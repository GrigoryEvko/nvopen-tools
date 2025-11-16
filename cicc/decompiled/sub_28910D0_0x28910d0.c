// Function: sub_28910D0
// Address: 0x28910d0
//
__int64 __fastcall sub_28910D0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // r13
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v6; // rbx
  int v7; // r14d
  __int64 v8; // rdx
  char v9; // al
  unsigned __int8 *v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v15; // [rsp+10h] [rbp-40h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v3 = 0;
  v4 = *(_QWORD *)(a3 + 80);
  v16 = a3 + 72;
  if ( v4 == a3 + 72 )
  {
    v11 = a1 + 32;
    v12 = a1 + 80;
    goto LABEL_23;
  }
  do
  {
    if ( !v4 )
      BUG();
    v5 = *(_QWORD *)(v4 + 32);
    v6 = v4 + 24;
    v7 = 0;
    if ( v4 + 24 != v5 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v8 = v5;
          v5 = *(_QWORD *)(v5 + 8);
          v9 = *(_BYTE *)(v8 - 24);
          v10 = (unsigned __int8 *)(v8 - 24);
          if ( v9 != 64 )
            break;
          sub_B43D60(v10);
          v7 = 1;
LABEL_6:
          if ( v6 == v5 )
            goto LABEL_10;
        }
        if ( v9 != 65 )
        {
          if ( v9 == 66 )
          {
            v7 |= sub_2A2DB90(v10);
          }
          else
          {
            v15 = v8;
            if ( (v9 == 61 || v9 == 62) && sub_B46500(v10) )
            {
              *(_WORD *)(v15 - 22) &= 0xFC7Fu;
              *(_BYTE *)(v15 + 48) = 1;
            }
          }
          goto LABEL_6;
        }
        v7 |= sub_2A2D840(v10);
        if ( v6 == v5 )
        {
LABEL_10:
          v3 |= v7;
          break;
        }
      }
    }
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v16 != v4 );
  v11 = a1 + 32;
  v12 = a1 + 80;
  if ( !v3 )
  {
LABEL_23:
    *(_QWORD *)(a1 + 8) = v11;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v12;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  memset((void *)a1, 0, 0x60u);
  *(_QWORD *)(a1 + 8) = v11;
  *(_DWORD *)(a1 + 16) = 2;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 56) = v12;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  return a1;
}
