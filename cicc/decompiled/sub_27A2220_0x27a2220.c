// Function: sub_27A2220
// Address: 0x27a2220
//
__int64 __fastcall sub_27A2220(__int64 *a1, int *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  int v7; // r8d
  __int64 v8; // rdi
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // r10d
  unsigned int i; // eax
  unsigned int *v15; // rsi
  __int64 v16; // r9
  unsigned int v17; // eax
  unsigned __int8 **v18; // rax
  unsigned int v19; // eax
  __int64 v20; // r15
  unsigned int v21; // r12d
  __int64 v22; // rax
  int v23; // esi
  __int64 v24; // rcx
  __int64 v25; // rdx
  int v26; // r10d
  __int64 v27; // r8
  unsigned int j; // eax
  unsigned int *v29; // rsi
  __int64 v30; // r9
  unsigned int v31; // eax
  unsigned __int8 **v32; // rax
  __int64 v34; // rdx
  unsigned int v35; // eax
  unsigned __int8 **v36; // [rsp+0h] [rbp-90h] BYREF
  __int64 v37; // [rsp+8h] [rbp-88h]
  _BYTE v38[32]; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int8 **v39; // [rsp+30h] [rbp-60h] BYREF
  __int64 v40; // [rsp+38h] [rbp-58h]
  _BYTE v41[80]; // [rsp+40h] [rbp-50h] BYREF

  v5 = a1[1];
  v6 = *a1;
  v7 = *(_DWORD *)(v5 + 24);
  v8 = *(_QWORD *)(v5 + 8);
  if ( !v7 )
  {
LABEL_7:
    v36 = (unsigned __int8 **)v38;
    v37 = 0x400000000LL;
LABEL_8:
    v18 = (unsigned __int8 **)v38;
    goto LABEL_9;
  }
  v9 = *((_QWORD *)a2 + 1);
  v10 = *a2;
  v11 = (unsigned int)(v7 - 1);
  v12 = (unsigned int)((0xBF58476D1CE4E5B9LL * v9) >> 31) ^ (484763065 * a2[2]);
  v13 = 1;
  for ( i = v11
          & (((0xBF58476D1CE4E5B9LL * (v12 | ((unsigned __int64)(unsigned int)(37 * *a2) << 32))) >> 31)
           ^ (484763065 * v12)); ; i = v11 & v17 )
  {
    v15 = (unsigned int *)(v8 + ((unsigned __int64)i << 6));
    v16 = *v15;
    if ( (_DWORD)v16 == v10 && *((_QWORD *)v15 + 1) == v9 )
      break;
    if ( (_DWORD)v16 == -1 && *((_QWORD *)v15 + 1) == -1 )
      goto LABEL_7;
    v17 = v13 + i;
    ++v13;
  }
  v34 = v15[6];
  v36 = (unsigned __int8 **)v38;
  v37 = 0x400000000LL;
  if ( !(_DWORD)v34 )
    goto LABEL_8;
  sub_27A1010((__int64)&v36, (__int64)(v15 + 4), v34, v9, v11, v16);
  v18 = v36;
LABEL_9:
  v19 = sub_27A2140(v6, *v18);
  v20 = *a1;
  v21 = v19;
  v22 = a1[1];
  v23 = *(_DWORD *)(v22 + 24);
  v24 = *(_QWORD *)(v22 + 8);
  if ( !v23 )
  {
LABEL_15:
    v39 = (unsigned __int8 **)v41;
    v40 = 0x400000000LL;
LABEL_16:
    v32 = (unsigned __int8 **)v41;
    goto LABEL_17;
  }
  v25 = *(_QWORD *)(a3 + 8);
  v26 = 1;
  v27 = (unsigned int)(v23 - 1);
  for ( j = v27
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)((0xBF58476D1CE4E5B9LL * v25) >> 31) ^ (484763065 * *(_DWORD *)(a3 + 8))
              | ((unsigned __int64)(unsigned int)(37 * *(_DWORD *)a3) << 32))) >> 31)
           ^ (484763065 * (((0xBF58476D1CE4E5B9LL * v25) >> 31) ^ (484763065 * *(_DWORD *)(a3 + 8))))); ; j = v27 & v31 )
  {
    v29 = (unsigned int *)(v24 + ((unsigned __int64)j << 6));
    v30 = *v29;
    if ( (_DWORD)v30 == *(_DWORD *)a3 && *((_QWORD *)v29 + 1) == v25 )
      break;
    if ( (_DWORD)v30 == -1 && *((_QWORD *)v29 + 1) == -1 )
      goto LABEL_15;
    v31 = v26 + j;
    ++v26;
  }
  v40 = 0x400000000LL;
  v35 = v29[6];
  v39 = (unsigned __int8 **)v41;
  if ( !v35 )
    goto LABEL_16;
  sub_27A1010((__int64)&v39, (__int64)(v29 + 4), v25, v24, v27, v30);
  v32 = v39;
LABEL_17:
  LOBYTE(v21) = v21 < (unsigned int)sub_27A2140(v20, *v32);
  if ( v39 != (unsigned __int8 **)v41 )
    _libc_free((unsigned __int64)v39);
  if ( v36 != (unsigned __int8 **)v38 )
    _libc_free((unsigned __int64)v36);
  return v21;
}
