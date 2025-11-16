// Function: sub_6EB360
// Address: 0x6eb360
//
__int64 __fastcall sub_6EB360(__int64 a1, __int64 a2, int a3, __int64 *a4)
{
  __int64 *v7; // rdx
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rcx
  _QWORD *v13; // rcx
  __int64 v14; // rax
  __int64 *v15; // [rsp+8h] [rbp-18h]

  v7 = a4;
  result = qword_4D03C50;
  if ( qword_4D03C50 && (*(_BYTE *)(qword_4D03C50 + 18LL) & 4) != 0 )
  {
    while ( *(_BYTE *)(a2 + 140) == 12 )
      a2 = *(_QWORD *)(a2 + 160);
    v10 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
    if ( v10 )
    {
      v11 = *(_QWORD *)(v10 + 24);
      if ( v11 )
      {
        if ( (*(_BYTE *)(v10 + 177) & 2) == 0 )
        {
          v12 = *(_QWORD *)(v11 + 88);
          if ( v12 )
          {
            *(_QWORD *)(a1 + 16) = v12;
            if ( (*(_BYTE *)(result + 17) & 2) != 0 )
              *(_BYTE *)(v12 + 193) |= 0x40u;
          }
          v13 = (_QWORD *)qword_4D03A78;
          if ( qword_4D03A78 )
          {
            qword_4D03A78 = *(_QWORD *)qword_4D03A78;
          }
          else
          {
            v15 = v7;
            v14 = sub_823970(24);
            v7 = v15;
            v13 = (_QWORD *)v14;
            result = qword_4D03C50;
          }
          *v13 = *(_QWORD *)(result + 32);
          *(_QWORD *)(qword_4D03C50 + 32LL) = v13;
          v13[1] = a1;
          result = *v7;
          v13[2] = *v7;
        }
      }
    }
  }
  else
  {
    result = sub_6EB2F0(a2, a3, (int)a4, 0);
    if ( result )
    {
      v9 = qword_4D03C50;
      *(_QWORD *)(a1 + 16) = result;
      if ( !v9 || (*(_BYTE *)(v9 + 17) & 2) != 0 )
        *(_BYTE *)(result + 193) |= 0x40u;
    }
  }
  return result;
}
