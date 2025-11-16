// Function: sub_22C6020
// Address: 0x22c6020
//
__int64 __fastcall sub_22C6020(unsigned __int8 *a1, __int64 a2, char a3)
{
  unsigned __int8 *v4; // rax
  char v5; // cl
  __int64 v6; // r8
  int v7; // edi
  __int64 result; // rax
  unsigned int v9; // edx
  __int64 v10; // r9
  unsigned __int8 *v11; // rsi
  unsigned int v12; // esi
  unsigned int v13; // eax
  int v14; // edx
  unsigned int v15; // edi
  _QWORD *v16; // r12
  int v17; // r11d
  _QWORD *v18; // r10
  char v19; // [rsp+7h] [rbp-59h] BYREF
  _QWORD *v20; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 *v22; // [rsp+20h] [rbp-40h]
  _QWORD v23[6]; // [rsp+30h] [rbp-30h] BYREF

  if ( a3 )
    v4 = sub_98ACB0(a1, 6u);
  else
    v4 = sub_BD4CB0(a1, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_96, (__int64)&v19);
  v21[0] = 0;
  v21[1] = 0;
  v22 = v4;
  if ( v4 != 0 && v4 + 4096 != 0 && v4 != (unsigned __int8 *)-8192LL )
    sub_BD73F0((__int64)v21);
  v5 = *(_BYTE *)(a2 + 8) & 1;
  if ( v5 )
  {
    v6 = a2 + 16;
    v7 = 1;
  }
  else
  {
    v12 = *(_DWORD *)(a2 + 24);
    v6 = *(_QWORD *)(a2 + 16);
    v7 = v12 - 1;
    if ( !v12 )
    {
      v13 = *(_DWORD *)(a2 + 8);
      ++*(_QWORD *)a2;
      v20 = 0;
      v14 = (v13 >> 1) + 1;
LABEL_15:
      v15 = 3 * v12;
      goto LABEL_16;
    }
  }
  result = (__int64)v22;
  v9 = v7 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v10 = v6 + 24LL * v9;
  v11 = *(unsigned __int8 **)(v10 + 16);
  if ( v11 == v22 )
    goto LABEL_9;
  v17 = 1;
  v18 = 0;
  while ( v11 != (unsigned __int8 *)-4096LL )
  {
    if ( v18 || v11 != (unsigned __int8 *)-8192LL )
      v10 = (__int64)v18;
    v9 = v7 & (v17 + v9);
    v11 = *(unsigned __int8 **)(v6 + 24LL * v9 + 16);
    if ( v22 == v11 )
      goto LABEL_9;
    ++v17;
    v18 = (_QWORD *)v10;
    v10 = v6 + 24LL * v9;
  }
  v13 = *(_DWORD *)(a2 + 8);
  if ( !v18 )
    v18 = (_QWORD *)v10;
  ++*(_QWORD *)a2;
  v20 = v18;
  v14 = (v13 >> 1) + 1;
  if ( !v5 )
  {
    v12 = *(_DWORD *)(a2 + 24);
    goto LABEL_15;
  }
  v15 = 6;
  v12 = 2;
LABEL_16:
  if ( v15 <= 4 * v14 )
  {
    v12 *= 2;
    goto LABEL_29;
  }
  if ( v12 - *(_DWORD *)(a2 + 12) - v14 <= v12 >> 3 )
  {
LABEL_29:
    sub_22C5A50(a2, v12);
    sub_22C3B00(a2, (__int64)v21, &v20);
    v13 = *(_DWORD *)(a2 + 8);
  }
  v16 = v20;
  v23[2] = -4096;
  v23[0] = 0;
  v23[1] = 0;
  *(_DWORD *)(a2 + 8) = (2 * (v13 >> 1) + 2) | v13 & 1;
  if ( v16[2] != -4096 )
    --*(_DWORD *)(a2 + 12);
  sub_D68D70(v23);
  sub_22BDC40(v16, (__int64)v22);
  result = (__int64)v22;
LABEL_9:
  if ( result != -4096 && result != 0 && result != -8192 )
    return sub_BD60C0(v21);
  return result;
}
