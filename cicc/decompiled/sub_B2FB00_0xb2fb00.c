// Function: sub_B2FB00
// Address: 0xb2fb00
//
__int64 __fastcall sub_B2FB00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int8 v5; // al
  unsigned __int8 v6; // al
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v10; // rdx

  if ( (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    v2 = sub_B91C10(a2, 20);
    v3 = v2;
    if ( v2 )
    {
      v4 = v2 - 16;
      v5 = *(_BYTE *)(v2 - 16);
      if ( (v5 & 2) != 0 )
      {
        sub_B91420(**(_QWORD **)(v3 - 32), 20);
        v6 = *(_BYTE *)(v3 - 16);
        if ( (v6 & 2) != 0 )
          goto LABEL_5;
      }
      else
      {
        sub_B91420(*(_QWORD *)(v4 - 8LL * ((v5 >> 2) & 0xF)), 20);
        v6 = *(_BYTE *)(v3 - 16);
        if ( (v6 & 2) != 0 )
        {
LABEL_5:
          v7 = *(_QWORD *)(v3 - 32);
LABEL_6:
          v8 = sub_B91420(*(_QWORD *)(v7 + 8), 20);
          *(_BYTE *)(a1 + 16) = 1;
          *(_QWORD *)a1 = v8;
          *(_QWORD *)(a1 + 8) = v10;
          return a1;
        }
      }
      v7 = v4 - 8LL * ((v6 >> 2) & 0xF);
      goto LABEL_6;
    }
  }
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
