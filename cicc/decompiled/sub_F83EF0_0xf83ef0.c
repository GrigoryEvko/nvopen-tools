// Function: sub_F83EF0
// Address: 0xf83ef0
//
__int64 __fastcall sub_F83EF0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r13
  unsigned int v4; // esi
  __int64 v5; // r12
  int v6; // edx
  unsigned __int8 *v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 v12; // rdx
  unsigned __int8 *v13; // r9
  unsigned __int8 *v14; // rdx
  int v15; // r10d
  int v16; // eax
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  __int64 v18; // [rsp+8h] [rbp-58h] BYREF
  void *v19; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v20[2]; // [rsp+18h] [rbp-48h] BYREF
  unsigned __int8 *v21; // [rsp+28h] [rbp-38h]
  unsigned __int8 v22; // [rsp+30h] [rbp-30h]

  v2 = a1 + 288;
  sub_F7D1E0((char *)&v18, a2);
  v21 = a2;
  v20[0] = 2;
  v20[1] = 0;
  if ( a2 + 4096 != 0 && a2 != 0 && a2 != (unsigned __int8 *)-8192LL )
    sub_BD73F0((__int64)v20);
  v4 = *(_DWORD *)(a1 + 312);
  v22 = 0;
  v19 = &unk_49E51C0;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 288);
    v17 = 0;
LABEL_6:
    v4 *= 2;
LABEL_7:
    sub_F83900(v2, v4);
    sub_F81D10(v2, (__int64)&v19, &v17);
    v5 = v17;
    v6 = *(_DWORD *)(a1 + 304) + 1;
    goto LABEL_8;
  }
  result = (__int64)v21;
  v10 = *(_QWORD *)(a1 + 296);
  v11 = (v4 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
  v12 = v10 + 48LL * v11;
  v13 = *(unsigned __int8 **)(v12 + 24);
  if ( v13 == v21 )
    goto LABEL_18;
  v15 = 1;
  v5 = 0;
  while ( v13 != (unsigned __int8 *)-4096LL )
  {
    if ( v5 || v13 != (unsigned __int8 *)-8192LL )
      v12 = v5;
    v11 = (v4 - 1) & (v15 + v11);
    v13 = *(unsigned __int8 **)(v10 + 48LL * v11 + 24);
    if ( v21 == v13 )
      goto LABEL_18;
    ++v15;
    v5 = v12;
    v12 = v10 + 48LL * v11;
  }
  v16 = *(_DWORD *)(a1 + 304);
  if ( !v5 )
    v5 = v12;
  ++*(_QWORD *)(a1 + 288);
  v6 = v16 + 1;
  v17 = v5;
  if ( 4 * (v16 + 1) >= 3 * v4 )
    goto LABEL_6;
  if ( v4 - *(_DWORD *)(a1 + 308) - v6 <= v4 >> 3 )
    goto LABEL_7;
LABEL_8:
  *(_DWORD *)(a1 + 304) = v6;
  if ( *(_QWORD *)(v5 + 24) != -4096 )
    --*(_DWORD *)(a1 + 308);
  if ( *(_BYTE *)(v5 + 32) )
  {
    *(_QWORD *)(v5 + 24) = 0;
    v7 = v21;
    if ( v21 )
    {
LABEL_12:
      *(_QWORD *)(v5 + 24) = v7;
      if ( v7 + 4096 != 0 && v7 != 0 && v7 != (unsigned __int8 *)-8192LL )
        sub_BD6050((unsigned __int64 *)(v5 + 8), v20[0] & 0xFFFFFFFFFFFFFFF8LL);
    }
  }
  else
  {
    v14 = *(unsigned __int8 **)(v5 + 24);
    v7 = v21;
    if ( v14 != v21 )
    {
      if ( v14 + 4096 != 0 && v14 != 0 && v14 != (unsigned __int8 *)-8192LL )
      {
        sub_BD60C0((_QWORD *)(v5 + 8));
        v7 = v21;
      }
      goto LABEL_12;
    }
  }
  result = v22;
  v9 = v18;
  *(_BYTE *)(v5 + 32) = v22;
  *(_QWORD *)(v5 + 40) = v9;
  if ( (_BYTE)result )
    return result;
  result = (__int64)v21;
LABEL_18:
  v19 = &unk_49DB368;
  if ( result != 0 && result != -4096 && result != -8192 )
    return sub_BD60C0(v20);
  return result;
}
