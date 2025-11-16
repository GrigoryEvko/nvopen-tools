// Function: sub_255AB60
// Address: 0x255ab60
//
__int64 __fastcall sub_255AB60(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // r15
  char v5; // al
  __int64 *v6; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  unsigned __int64 v12; // r8
  unsigned __int64 i; // r12
  unsigned __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // [rsp+8h] [rbp-68h]
  unsigned int v18; // [rsp+14h] [rbp-5Ch]
  unsigned __int64 v19; // [rsp+18h] [rbp-58h]
  int v20; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v21[8]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
    v4 = *(_QWORD *)(v4 + 24);
  v5 = *(_BYTE *)(a1 + 97);
  if ( (v5 & 3) == 3 )
  {
    sub_255AA10(v21, 0);
    v18 = v21[0];
LABEL_15:
    v9 = a1 + 72;
    v17 = (__int64 *)(a1 + 72);
    sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)dword_438A680, 3);
    v10 = v18 >> 6;
    if ( (((unsigned __int8)v18 | (unsigned __int8)(v10 | (v18 >> 2) | (v18 >> 4))) & 2) != 0 )
      goto LABEL_7;
    goto LABEL_9;
  }
  if ( (v5 & 2) == 0 )
  {
    if ( (v5 & 1) == 0 )
    {
      v17 = (__int64 *)(a1 + 72);
      sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)dword_438A680, 3);
      v18 = 255;
      goto LABEL_7;
    }
    sub_255AA10(v21, 2u);
    v18 = v21[0];
    goto LABEL_15;
  }
  v9 = a1 + 72;
  v17 = (__int64 *)(a1 + 72);
  sub_2515E10(a2, (__int64 *)(a1 + 72), (__int64)dword_438A680, 3);
  v18 = 85;
LABEL_9:
  if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
  {
    sub_B2C6D0(v4, v9, v10, v11);
    v12 = *(_QWORD *)(v4 + 96);
    v19 = v12 + 40LL * *(_QWORD *)(v4 + 104);
    if ( (*(_BYTE *)(v4 + 2) & 1) != 0 )
    {
      sub_B2C6D0(v4, v9, v15, v16);
      v12 = *(_QWORD *)(v4 + 96);
    }
  }
  else
  {
    v12 = *(_QWORD *)(v4 + 96);
    v19 = v12 + 40LL * *(_QWORD *)(v4 + 104);
  }
  for ( i = v12; v19 != i; i += 40LL )
  {
    v14 = i;
    v20 = 77;
    sub_250D230((unsigned __int64 *)v21, v14, 6, 0);
    sub_2515E10(a2, v21, (__int64)&v20, 1);
  }
LABEL_7:
  v6 = (__int64 *)sub_B2BE50(v4);
  v21[0] = sub_A77AB0(v6, v18);
  return sub_2516380(a2, v17, (__int64)v21, 1, 0);
}
