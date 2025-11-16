// Function: sub_735E40
// Address: 0x735e40
//
void __fastcall sub_735E40(__int64 a1, int a2)
{
  __int64 v2; // rbx
  _BYTE *v3; // rdx
  unsigned __int64 v4; // rax
  int v5; // ecx
  __int64 v6; // rbx
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = a2;
  v3 = sub_735B90(a2, a1, v12);
  v4 = (unsigned __int8)v3[28];
  if ( (unsigned __int8)v4 <= 7u )
  {
    v8 = 201;
    v5 = dword_4F04C64;
    if ( _bittest64(&v8, v4) )
      goto LABEL_15;
  }
  else
  {
    v5 = dword_4F04C64;
  }
  if ( (_DWORD)v2 == -1 )
  {
    if ( v5 >= 0 )
    {
      v6 = qword_4F04C68[0];
      goto LABEL_6;
    }
LABEL_30:
    BUG();
  }
  if ( (int)v2 > v5 )
    goto LABEL_30;
  v6 = qword_4F04C68[0] + 776 * v2;
LABEL_6:
  if ( *(_BYTE *)(v6 + 4) != 1 )
  {
    v7 = sub_732EF0(v6);
    v3 = v7;
    if ( v7 )
    {
      if ( *(_BYTE *)(a1 + 136) > 2u )
      {
        if ( *((_QWORD *)v7 + 15) )
          *(_QWORD *)(*(_QWORD *)(v6 + 288) + 112LL) = a1;
        else
          *((_QWORD *)v7 + 15) = a1;
        *(_QWORD *)(v6 + 288) = a1;
LABEL_13:
        sub_72EE40(a1, 7u, (__int64)v3);
LABEL_20:
        *(_QWORD *)(a1 + 112) = 0;
        return;
      }
LABEL_15:
      v9 = *((_QWORD *)v3 + 14);
      v10 = v12[0];
      if ( v9 )
      {
        if ( !v12[0] )
        {
          do
          {
            v11 = v9;
            v9 = *(_QWORD *)(v9 + 112);
          }
          while ( v9 );
          *(_QWORD *)(v11 + 112) = a1;
LABEL_19:
          if ( *(_QWORD *)(a1 + 40) || (*(_BYTE *)(a1 + 89) & 2) != 0 )
            goto LABEL_20;
          goto LABEL_13;
        }
        *(_QWORD *)(*(_QWORD *)(v12[0] + 40) + 112LL) = a1;
      }
      else
      {
        *((_QWORD *)v3 + 14) = a1;
        if ( !v10 )
          goto LABEL_19;
      }
      *(_QWORD *)(v10 + 40) = a1;
      goto LABEL_19;
    }
  }
}
