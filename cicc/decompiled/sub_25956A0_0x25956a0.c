// Function: sub_25956A0
// Address: 0x25956a0
//
__int64 __fastcall sub_25956A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r12
  char v6; // al
  char v7; // r13
  int v8; // r14d
  __int64 v9; // rsi
  __int64 v10; // rdi
  char v11; // al
  char v12; // al
  bool v13; // zf
  char v15; // al
  unsigned __int64 v16[6]; // [rsp+0h] [rbp-30h] BYREF

  v3 = sub_25096F0((_QWORD *)(a1 + 72));
  sub_250D230(v16, v3, 4, 0);
  v4 = sub_25952D0(a2, v16[0], v16[1], a1, 0, 0, 1);
  if ( v4 )
  {
    v5 = v4;
    v6 = *(_BYTE *)(v4 + 136);
    v7 = *(_BYTE *)(a1 + 136);
    v8 = *(_DWORD *)(a1 + 160);
    if ( v6 )
    {
      v6 = *(_BYTE *)(a1 + 136);
    }
    else
    {
      v9 = v5 + 144;
      v10 = a1 + 144;
      if ( !v7 )
      {
        sub_2561130(v10, v9);
LABEL_6:
        v11 = *(_BYTE *)(v5 + 136) & *(_BYTE *)(a1 + 136);
        *(_BYTE *)(a1 + 136) = v11;
        v12 = *(_BYTE *)(a1 + 96) | v11;
        if ( v12 )
        {
LABEL_10:
          v13 = *(_DWORD *)(a1 + 160) == v8;
          *(_BYTE *)(a1 + 136) = v12;
          return (unsigned __int8)(!v13 | v7 ^ v12) ^ 1u;
        }
LABEL_7:
        sub_2577D20(a1 + 144, a1 + 104);
        v12 = *(_BYTE *)(a1 + 96) | *(_BYTE *)(a1 + 136);
        goto LABEL_10;
      }
      if ( v9 != v10 )
      {
        sub_255D9B0(v10, v9);
        goto LABEL_6;
      }
      *(_BYTE *)(a1 + 136) = 0;
    }
    v12 = *(_BYTE *)(a1 + 96) | v6;
    if ( v12 )
      goto LABEL_10;
    goto LABEL_7;
  }
  v15 = *(_BYTE *)(a1 + 96);
  *(_BYTE *)(a1 + 176) = 1;
  *(_BYTE *)(a1 + 136) = v15;
  sub_255D9B0(a1 + 144, a1 + 104);
  return 0;
}
