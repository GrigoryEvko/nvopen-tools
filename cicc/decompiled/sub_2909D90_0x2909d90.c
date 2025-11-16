// Function: sub_2909D90
// Address: 0x2909d90
//
__int64 __fastcall sub_2909D90(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4)
{
  __int64 v7; // r13
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  int v13; // eax
  char *v14; // rcx
  __int64 v15; // rdi
  __int64 *v16; // r12
  char *v17; // rsi
  _QWORD *v18; // rax
  int v19; // r8d
  __int64 v20; // rax
  __int64 *v21; // r14
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 *v24; // r10
  int v25; // r11d
  unsigned int v26; // eax
  __int64 *v27; // rdi
  __int64 v28; // rcx
  unsigned int v29; // esi
  int v30; // edx
  __int64 v31; // r15
  __int64 v32; // rax
  int v34; // eax
  int v35; // ecx
  int v36; // r8d
  __int64 v37; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v38; // [rsp+8h] [rbp-68h] BYREF
  __int64 v39; // [rsp+10h] [rbp-60h] BYREF
  char *v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+28h] [rbp-48h]
  __int64 *v43; // [rsp+30h] [rbp-40h] BYREF
  __int64 v44; // [rsp+38h] [rbp-38h]
  _BYTE v45[48]; // [rsp+40h] [rbp-30h] BYREF

  v37 = *(_QWORD *)(a1 + 40);
  v7 = sub_2909AD0(a2 + 144, &v37);
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  sub_C7D6A0(0, 0, 8);
  v12 = *(unsigned int *)(v7 + 24);
  LODWORD(v42) = v12;
  if ( (_DWORD)v12 )
  {
    v40 = (char *)sub_C7D670(8 * v12, 8);
    v41 = *(_QWORD *)(v7 + 16);
    memcpy(v40, *(const void **)(v7 + 8), 8LL * (unsigned int)v42);
  }
  else
  {
    v40 = 0;
    v41 = 0;
  }
  v44 = 0;
  v43 = (__int64 *)v45;
  if ( *(_DWORD *)(v7 + 40) )
    sub_28FEFA0((__int64)&v43, v7 + 32, v8, v9, v10, v11);
  sub_2909780(
    (_QWORD *)(*(_QWORD *)(v37 + 48) & 0xFFFFFFFFFFFFFFF8LL),
    (_QWORD *)(*(_QWORD *)(a1 + 24) & 0xFFFFFFFFFFFFFFF8LL),
    (__int64)&v39,
    a4);
  v38 = (__int64 *)a1;
  if ( !(_DWORD)v42 )
  {
LABEL_40:
    v16 = v43;
    v20 = (unsigned int)v44;
    goto LABEL_10;
  }
  v13 = (v42 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v14 = &v40[8 * v13];
  v15 = *(_QWORD *)v14;
  if ( a1 != *(_QWORD *)v14 )
  {
    v35 = 1;
    while ( v15 != -4096 )
    {
      v36 = v35 + 1;
      v13 = (v42 - 1) & (v35 + v13);
      v14 = &v40[8 * v13];
      v15 = *(_QWORD *)v14;
      if ( a1 == *(_QWORD *)v14 )
        goto LABEL_7;
      v35 = v36;
    }
    goto LABEL_40;
  }
LABEL_7:
  *(_QWORD *)v14 = -8192;
  v16 = v43;
  LODWORD(v41) = v41 - 1;
  ++HIDWORD(v41);
  v17 = (char *)&v43[(unsigned int)v44];
  v18 = sub_28FEBC0(v43, (__int64)v17, (__int64 *)&v38);
  if ( v18 + 1 != (_QWORD *)v17 )
  {
    memmove(v18, v18 + 1, v17 - (char *)(v18 + 1));
    v19 = v44;
    v16 = v43;
  }
  v20 = (unsigned int)(v19 - 1);
  LODWORD(v44) = v19 - 1;
LABEL_10:
  v21 = &v16[v20];
  if ( v21 == v16 )
    goto LABEL_24;
  do
  {
    while ( 1 )
    {
      v29 = *(_DWORD *)(a3 + 24);
      if ( !v29 )
      {
        ++*(_QWORD *)a3;
        v38 = 0;
LABEL_16:
        v29 *= 2;
LABEL_17:
        sub_CE2A30(a3, v29);
        sub_DA5B20(a3, v16, &v38);
        v24 = v38;
        v30 = *(_DWORD *)(a3 + 16) + 1;
        goto LABEL_18;
      }
      v22 = v29 - 1;
      v23 = *(_QWORD *)(a3 + 8);
      v24 = 0;
      v25 = 1;
      v26 = v22 & (((unsigned int)*v16 >> 9) ^ ((unsigned int)*v16 >> 4));
      v27 = (__int64 *)(v23 + 8LL * v26);
      v28 = *v27;
      if ( *v27 != *v16 )
        break;
LABEL_13:
      if ( v21 == ++v16 )
        goto LABEL_23;
    }
    while ( v28 != -4096 )
    {
      if ( v24 || v28 != -8192 )
        v27 = v24;
      v26 = v22 & (v25 + v26);
      v28 = *(_QWORD *)(v23 + 8LL * v26);
      if ( *v16 == v28 )
        goto LABEL_13;
      ++v25;
      v24 = v27;
      v27 = (__int64 *)(v23 + 8LL * v26);
    }
    v34 = *(_DWORD *)(a3 + 16);
    if ( !v24 )
      v24 = v27;
    ++*(_QWORD *)a3;
    v30 = v34 + 1;
    v38 = v24;
    if ( 4 * (v34 + 1) >= 3 * v29 )
      goto LABEL_16;
    if ( v29 - *(_DWORD *)(a3 + 20) - v30 <= v29 >> 3 )
      goto LABEL_17;
LABEL_18:
    *(_DWORD *)(a3 + 16) = v30;
    if ( *v24 != -4096 )
      --*(_DWORD *)(a3 + 20);
    v31 = *v16;
    *v24 = *v16;
    v32 = *(unsigned int *)(a3 + 40);
    if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 44) )
    {
      sub_C8D5F0(a3 + 32, (const void *)(a3 + 48), v32 + 1, 8u, v23, v22);
      v32 = *(unsigned int *)(a3 + 40);
    }
    ++v16;
    *(_QWORD *)(*(_QWORD *)(a3 + 32) + 8 * v32) = v31;
    ++*(_DWORD *)(a3 + 40);
  }
  while ( v21 != v16 );
LABEL_23:
  v16 = v43;
LABEL_24:
  if ( v16 != (__int64 *)v45 )
    _libc_free((unsigned __int64)v16);
  return sub_C7D6A0((__int64)v40, 8LL * (unsigned int)v42, 8);
}
