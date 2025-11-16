// Function: sub_255AD80
// Address: 0x255ad80
//
__int64 __fastcall sub_255AD80(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r12
  char v3; // al
  unsigned int v4; // r15d
  __int64 *v5; // rax
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rbx
  unsigned __int8 *v14; // rdx
  int v15; // eax
  __int64 *v16; // [rsp+8h] [rbp-68h]
  __int64 v17; // [rsp+18h] [rbp-58h]
  int v18; // [rsp+18h] [rbp-58h]
  unsigned __int8 *v19; // [rsp+18h] [rbp-58h]
  int v20; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v21[8]; // [rsp+30h] [rbp-40h] BYREF

  v2 = (unsigned __int8 *)(*(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL);
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v2 = (unsigned __int8 *)*((_QWORD *)v2 + 3);
  v3 = *(_BYTE *)(a1 + 97);
  if ( (v3 & 3) == 3 )
  {
    sub_255AA10(v21, 0);
    v4 = v21[0];
  }
  else if ( (v3 & 2) != 0 )
  {
    sub_255AA10(v21, 1u);
    v4 = v21[0];
  }
  else
  {
    if ( (v3 & 1) == 0 )
    {
      v16 = (__int64 *)(a1 + 72);
      v4 = 255;
      sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)dword_438A680, 3);
      goto LABEL_7;
    }
    sub_255AA10(v21, 2u);
    v4 = v21[0];
  }
  v16 = (__int64 *)(a1 + 72);
  sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)dword_438A680, 3);
  if ( (((unsigned __int8)v4 | (unsigned __int8)((v4 >> 6) | (v4 >> 2) | (v4 >> 4))) & 2) != 0 )
    goto LABEL_7;
  v7 = *v2;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v2);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (v2[7] & 0x80u) != 0 )
  {
    v9 = sub_BD2BC0((__int64)v2);
    v17 = v10 + v9;
    if ( (v2[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v17 >> 4) )
        goto LABEL_32;
    }
    else if ( (unsigned int)((v17 - sub_BD2BC0((__int64)v2)) >> 4) )
    {
      if ( (v2[7] & 0x80u) != 0 )
      {
        v18 = *(_DWORD *)(sub_BD2BC0((__int64)v2) + 8);
        if ( (v2[7] & 0x80u) == 0 )
          BUG();
        v11 = sub_BD2BC0((__int64)v2);
        v8 -= 32LL * (unsigned int)(*(_DWORD *)(v11 + v12 - 4) - v18);
        goto LABEL_18;
      }
LABEL_32:
      BUG();
    }
  }
LABEL_18:
  v19 = &v2[v8];
  v13 = (__int64)&v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
  if ( (unsigned __int8 *)v13 != v19 )
  {
    do
    {
      v20 = 77;
      v15 = sub_BD2910(v13);
      if ( (v2[7] & 0x40) != 0 )
        v14 = (unsigned __int8 *)*((_QWORD *)v2 - 1);
      else
        v14 = &v2[-32 * (*((_DWORD *)v2 + 1) & 0x7FFFFFF)];
      v21[1] = 0;
      v13 += 32;
      v21[0] = (unsigned __int64)&v14[32 * v15] | 3;
      nullsub_1518();
      sub_2515E10(a2, v21, (__int64)&v20, 1);
    }
    while ( v19 != (unsigned __int8 *)v13 );
  }
LABEL_7:
  v5 = (__int64 *)sub_BD5C60((__int64)v2);
  v21[0] = sub_A77AB0(v5, v4);
  return sub_2516380(a2, v16, (__int64)v21, 1, 0);
}
