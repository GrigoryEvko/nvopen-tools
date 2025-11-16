// Function: sub_35503A0
// Address: 0x35503a0
//
__int64 __fastcall sub_35503A0(__int64 a1, _QWORD *a2)
{
  int v2; // r9d
  int v3; // r14d
  __int64 **v4; // rbx
  __int64 **v5; // r12
  __int64 result; // rax
  __int64 v7; // r15
  __int64 *v8; // r8
  __int64 v9; // rdi
  int v10; // r10d
  int v11; // r10d
  __int64 v12; // rcx
  __int64 v13; // r13
  unsigned int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rcx
  unsigned int v18; // edx
  __int64 **v19; // r9
  __int64 *v20; // rsi
  __int64 v21; // r10
  _QWORD *v22; // rcx
  char v23; // al
  __int64 v24; // rdi
  __int64 *v25; // r8
  bool v26; // zf
  _QWORD *v27; // rax
  int v28; // edx
  unsigned int v29; // esi
  int v30; // edx
  _QWORD *v31; // rdx
  unsigned int v32; // esi
  __int64 v33; // r14
  __int64 v34; // rcx
  int v35; // r8d
  __int64 *v36; // rdi
  __int64 v37; // r11
  unsigned int v38; // edx
  __int64 *v39; // rax
  __int64 v40; // r9
  int v41; // r9d
  int v42; // r11d
  int v43; // edx
  int v44; // edx
  __int64 **v45; // [rsp+0h] [rbp-80h]
  __int64 *v46; // [rsp+0h] [rbp-80h]
  __int64 *v47; // [rsp+8h] [rbp-78h]
  _QWORD *v48; // [rsp+8h] [rbp-78h]
  __int64 **v50; // [rsp+18h] [rbp-68h]
  unsigned int v51; // [rsp+28h] [rbp-58h] BYREF
  unsigned int v52; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v53; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v54; // [rsp+38h] [rbp-48h] BYREF
  _QWORD *v55; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v56[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = 0;
  v3 = 0;
  v4 = (__int64 **)a2[2];
  v5 = (__int64 **)a2[4];
  result = a2[6];
  v7 = a2[5];
  v50 = (__int64 **)result;
  if ( (__int64 **)result == v4 )
    return result;
  do
  {
    v8 = *v4;
    v9 = **v4;
    v10 = *(_DWORD *)(v9 + 40);
    v53 = v9;
    v11 = v10 & 0xFFFFFF;
    if ( !v11 )
      goto LABEL_10;
    v12 = *(_QWORD *)(v9 + 32);
    v13 = 0;
    v14 = 0;
    while ( 1 )
    {
      result = v12 + v13;
      if ( *(_BYTE *)(v12 + v13) )
        goto LABEL_5;
      if ( (*(_BYTE *)(result + 3) & 0x10) == 0 )
        break;
      if ( (*(_WORD *)(result + 2) & 0xFF0) != 0 )
      {
        v15 = sub_2E89F40(v9, v14);
        v16 = *(_QWORD *)(v53 + 32);
        result = 5LL * v15;
        v2 = *(_DWORD *)(v16 + 8 * result + 8);
        v3 = *(_DWORD *)(v16 + v13 + 8);
        goto LABEL_10;
      }
LABEL_5:
      ++v14;
      v13 += 40;
      if ( v11 == v14 )
        goto LABEL_10;
    }
    if ( *(_DWORD *)(v12 + v13 + 8) != v2 )
      goto LABEL_5;
    v17 = *(_QWORD *)(a1 + 4024);
    result = *(unsigned int *)(a1 + 4040);
    if ( !(_DWORD)result )
      goto LABEL_18;
    v18 = (result - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v19 = (__int64 **)(v17 + 24LL * v18);
    v20 = *v19;
    if ( v8 != *v19 )
    {
      v41 = 1;
      while ( v20 != (__int64 *)-4096LL )
      {
        v42 = v41 + 1;
        v18 = (result - 1) & (v18 + v41);
        v19 = (__int64 **)(v17 + 24LL * v18);
        v20 = *v19;
        if ( v8 == *v19 )
          goto LABEL_16;
        v41 = v42;
      }
      goto LABEL_18;
    }
LABEL_16:
    result = v17 + 24 * result;
    if ( v19 == (__int64 **)result )
      goto LABEL_18;
    v21 = *(_QWORD *)(a1 + 16);
    result = *(_QWORD *)(*(_QWORD *)v21 + 824LL);
    if ( (__int64 (*)())result == sub_2FDC6B0 )
      goto LABEL_18;
    v45 = v19;
    v47 = v8;
    result = ((__int64 (__fastcall *)(__int64, __int64, unsigned int *, unsigned int *))result)(v21, v9, &v51, &v52);
    if ( !(_BYTE)result )
      goto LABEL_18;
    v54 = sub_2E7B2C0(*(_QWORD **)(a1 + 32), v53);
    sub_2EAB0C0(v54[4] + 40LL * v51, v3);
    v22 = v54;
    *(_QWORD *)(v54[4] + 40LL * v52 + 24) = *(_QWORD *)(*(_QWORD *)(v53 + 32) + 40LL * v52 + 24) - (_QWORD)v45[2];
    *((_BYTE *)v47 + 254) |= 8u;
    *v47 = (__int64)v22;
    v23 = sub_3547A70(a1 + 936, (__int64 *)&v54, &v55);
    v24 = a1 + 936;
    v25 = v47;
    v26 = v23 == 0;
    v27 = v55;
    if ( v26 )
    {
      v56[0] = v55;
      v28 = *(_DWORD *)(a1 + 952);
      v29 = *(_DWORD *)(a1 + 960);
      ++*(_QWORD *)(a1 + 936);
      v30 = v28 + 1;
      if ( 4 * v30 >= 3 * v29 )
      {
        v46 = v47;
        v29 *= 2;
      }
      else
      {
        if ( v29 - *(_DWORD *)(a1 + 956) - v30 > v29 >> 3 )
          goto LABEL_23;
        v46 = v47;
      }
      sub_2F960A0(v24, v29);
      sub_3547A70(v24, (__int64 *)&v54, v56);
      v25 = v46;
      v30 = *(_DWORD *)(a1 + 952) + 1;
      v27 = (_QWORD *)v56[0];
LABEL_23:
      *(_DWORD *)(a1 + 952) = v30;
      if ( *v27 != -4096 )
        --*(_DWORD *)(a1 + 956);
      v31 = v54;
      v27[1] = 0;
      *v27 = v31;
    }
    v27[1] = v25;
    v48 = v54;
    v32 = *(_DWORD *)(a1 + 4072);
    v33 = a1 + 4048;
    if ( !v32 )
    {
      v56[0] = 0;
      ++*(_QWORD *)(a1 + 4048);
LABEL_47:
      v32 *= 2;
LABEL_48:
      sub_2E48800(v33, v32);
      sub_3547B30(v33, &v53, v56);
      v34 = v53;
      v44 = *(_DWORD *)(a1 + 4064) + 1;
      v39 = (__int64 *)v56[0];
      goto LABEL_43;
    }
    v34 = v53;
    v35 = 1;
    v36 = 0;
    v37 = *(_QWORD *)(a1 + 4056);
    v38 = (v32 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
    v39 = (__int64 *)(v37 + 16LL * v38);
    v40 = *v39;
    if ( v53 == *v39 )
      goto LABEL_28;
    while ( v40 != -4096 )
    {
      if ( !v36 && v40 == -8192 )
        v36 = v39;
      v38 = (v32 - 1) & (v35 + v38);
      v39 = (__int64 *)(v37 + 16LL * v38);
      v40 = *v39;
      if ( v53 == *v39 )
        goto LABEL_28;
      ++v35;
    }
    if ( v36 )
      v39 = v36;
    ++*(_QWORD *)(a1 + 4048);
    v43 = *(_DWORD *)(a1 + 4064);
    v56[0] = v39;
    v44 = v43 + 1;
    if ( 4 * v44 >= 3 * v32 )
      goto LABEL_47;
    if ( v32 - *(_DWORD *)(a1 + 4068) - v44 <= v32 >> 3 )
      goto LABEL_48;
LABEL_43:
    *(_DWORD *)(a1 + 4064) = v44;
    if ( *v39 != -4096 )
      --*(_DWORD *)(a1 + 4068);
    *v39 = v34;
    v39[1] = 0;
LABEL_28:
    result = (__int64)(v39 + 1);
    *(_QWORD *)result = v48;
LABEL_18:
    v3 = 0;
    v2 = 0;
LABEL_10:
    if ( v5 == ++v4 )
    {
      v4 = *(__int64 ***)(v7 + 8);
      v7 += 8;
      v5 = v4 + 64;
    }
  }
  while ( v4 != v50 );
  return result;
}
