// Function: sub_3167600
// Address: 0x3167600
//
__int64 *__fastcall sub_3167600(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rcx
  unsigned int v6; // esi
  __int64 v7; // rdx
  unsigned int v8; // r10d
  unsigned int v9; // edi
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  _QWORD *v14; // r8
  char v15; // r15
  __int64 *v16; // r14
  int v17; // ecx
  _QWORD *v18; // rax
  __int64 v19; // r11
  _DWORD *v20; // rax
  __int64 v21; // r13
  unsigned int v22; // esi
  __int64 v23; // r10
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // rcx
  unsigned int v29; // edx
  __int64 *v30; // rax
  __int64 v31; // r9
  unsigned int v32; // esi
  __int64 v33; // r8
  unsigned int v34; // edx
  __int64 *result; // rax
  __int64 v36; // rcx
  int v37; // eax
  __int64 v38; // rax
  __int64 v39; // r10
  __int64 v40; // rdi
  int v41; // edx
  int v42; // eax
  int v43; // r10d
  __int64 *v44; // rdi
  int v45; // eax
  int v46; // edx
  __int64 v47; // rdi
  int v48; // eax
  int v49; // edx
  __int64 v50; // rcx
  int v51; // r11d
  __int64 *v52; // r10
  int v53; // eax
  int v54; // edx
  __int64 v55; // rcx
  int v56; // r9d
  int v57; // [rsp+8h] [rbp-68h]
  int v58; // [rsp+8h] [rbp-68h]
  _QWORD *v59; // [rsp+10h] [rbp-60h]
  int v60; // [rsp+10h] [rbp-60h]
  _QWORD *v61; // [rsp+10h] [rbp-60h]
  _QWORD *v62; // [rsp+10h] [rbp-60h]
  unsigned int v63; // [rsp+18h] [rbp-58h]
  __int64 v64; // [rsp+18h] [rbp-58h]
  int v65; // [rsp+18h] [rbp-58h]
  __int64 v66; // [rsp+18h] [rbp-58h]
  _QWORD *v67; // [rsp+18h] [rbp-58h]
  __int64 *v68; // [rsp+28h] [rbp-48h] BYREF
  __int64 v69; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v70; // [rsp+38h] [rbp-38h]

  v4 = a1[1];
  v5 = *a1;
  v6 = *(_DWORD *)(v4 + 48);
  v7 = *(_QWORD *)(v4 + 32);
  if ( v6 )
  {
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_3:
      v12 = v4 + 24;
      v13 = *(_QWORD *)(v5 + 32) + 48LL * *((unsigned int *)v10 + 2);
      v14 = *(_QWORD **)(v13 + 8);
      v15 = *(_BYTE *)(v13 + 32);
      v16 = *(__int64 **)(v13 + 40);
      v17 = *(_DWORD *)(v13 + 24);
      v68 = (__int64 *)a2;
      goto LABEL_4;
    }
    v37 = 1;
    while ( v11 != -4096 )
    {
      v56 = v37 + 1;
      v9 = v8 & (v37 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v37 = v56;
    }
  }
  v12 = v4 + 24;
  v38 = *(_QWORD *)(v5 + 32) + 48LL * *(unsigned int *)(v7 + 16LL * v6 + 8);
  v14 = *(_QWORD **)(v38 + 8);
  v15 = *(_BYTE *)(v38 + 32);
  v16 = *(__int64 **)(v38 + 40);
  v17 = *(_DWORD *)(v38 + 24);
  v68 = (__int64 *)a2;
  if ( !v6 )
  {
    v69 = 0;
    ++*(_QWORD *)(v4 + 24);
    goto LABEL_19;
  }
  v8 = v6 - 1;
LABEL_4:
  v63 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (_QWORD *)(v7 + 16LL * v63);
  v19 = *v18;
  if ( a2 == *v18 )
  {
LABEL_5:
    v20 = v18 + 1;
    goto LABEL_6;
  }
  v60 = 1;
  v40 = 0;
  while ( v19 != -4096 )
  {
    if ( v19 == -8192 && !v40 )
      v40 = (__int64)v18;
    v63 = v8 & (v60 + v63);
    v18 = (_QWORD *)(v7 + 16LL * v63);
    v19 = *v18;
    if ( a2 == *v18 )
      goto LABEL_5;
    ++v60;
  }
  if ( !v40 )
    v40 = (__int64)v18;
  v69 = v40;
  v42 = *(_DWORD *)(v4 + 40);
  ++*(_QWORD *)(v4 + 24);
  v41 = v42 + 1;
  if ( 4 * (v42 + 1) < 3 * v6 )
  {
    v39 = a2;
    if ( v6 - *(_DWORD *)(v4 + 44) - v41 <= v6 >> 3 )
    {
      v58 = v17;
      v61 = v14;
      v66 = v12;
      sub_D39D40(v12, v6);
      sub_22B1A50(v66, (__int64 *)&v68, &v69);
      v39 = (__int64)v68;
      v40 = v69;
      v17 = v58;
      v14 = v61;
      v41 = *(_DWORD *)(v4 + 40) + 1;
    }
    goto LABEL_27;
  }
LABEL_19:
  v57 = v17;
  v59 = v14;
  v64 = v12;
  sub_D39D40(v12, 2 * v6);
  sub_22B1A50(v64, (__int64 *)&v68, &v69);
  v39 = (__int64)v68;
  v40 = v69;
  v14 = v59;
  v17 = v57;
  v41 = *(_DWORD *)(v4 + 40) + 1;
LABEL_27:
  *(_DWORD *)(v4 + 40) = v41;
  if ( *(_QWORD *)v40 != -4096 )
    --*(_DWORD *)(v4 + 44);
  *(_QWORD *)v40 = v39;
  v20 = (_DWORD *)(v40 + 8);
  *(_DWORD *)(v40 + 8) = 0;
LABEL_6:
  *v20 = v17;
  v21 = a1[1];
  v69 = a2;
  LOBYTE(v70) = v15;
  v22 = *(_DWORD *)(v21 + 80);
  if ( !v22 )
  {
    v68 = 0;
    ++*(_QWORD *)(v21 + 56);
LABEL_66:
    v62 = v14;
    v22 *= 2;
LABEL_67:
    sub_3167420(v21 + 56, v22);
    sub_3163250(v21 + 56, &v69, &v68);
    v50 = v69;
    v47 = (__int64)v68;
    v14 = v62;
    v49 = *(_DWORD *)(v21 + 72) + 1;
    goto LABEL_45;
  }
  v23 = *(_QWORD *)(v21 + 64);
  v24 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( a2 == *v25 )
    goto LABEL_8;
  v65 = 1;
  v47 = 0;
  while ( v26 != -4096 )
  {
    if ( v47 || v26 != -8192 )
      v25 = (__int64 *)v47;
    v24 = (v22 - 1) & (v65 + v24);
    v26 = *(_QWORD *)(v23 + 16LL * v24);
    if ( a2 == v26 )
      goto LABEL_8;
    ++v65;
    v47 = (__int64)v25;
    v25 = (__int64 *)(v23 + 16LL * v24);
  }
  if ( !v47 )
    v47 = (__int64)v25;
  v68 = (__int64 *)v47;
  v48 = *(_DWORD *)(v21 + 72);
  ++*(_QWORD *)(v21 + 56);
  v49 = v48 + 1;
  if ( 4 * (v48 + 1) >= 3 * v22 )
    goto LABEL_66;
  v50 = a2;
  if ( v22 - *(_DWORD *)(v21 + 76) - v49 <= v22 >> 3 )
  {
    v62 = v14;
    goto LABEL_67;
  }
LABEL_45:
  *(_DWORD *)(v21 + 72) = v49;
  if ( *(_QWORD *)v47 != -4096 )
    --*(_DWORD *)(v21 + 76);
  *(_QWORD *)v47 = v50;
  *(_BYTE *)(v47 + 8) = (_BYTE)v70;
  v21 = a1[1];
LABEL_8:
  if ( v16 )
    v16 = (__int64 *)((char *)v16 + (1LL << v15));
  v69 = a2;
  v70 = v16;
  v27 = *(_DWORD *)(v21 + 112);
  if ( !v27 )
  {
    v68 = 0;
    ++*(_QWORD *)(v21 + 88);
LABEL_63:
    v67 = v14;
    v27 *= 2;
LABEL_64:
    sub_9BBF00(v21 + 88, v27);
    sub_28EE430(v21 + 88, &v69, &v68);
    v55 = v69;
    v52 = v68;
    v14 = v67;
    v54 = *(_DWORD *)(v21 + 104) + 1;
    goto LABEL_54;
  }
  v28 = *(_QWORD *)(v21 + 96);
  v29 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v30 = (__int64 *)(v28 + 16LL * v29);
  v31 = *v30;
  if ( a2 == *v30 )
    goto LABEL_12;
  v51 = 1;
  v52 = 0;
  while ( v31 != -4096 )
  {
    if ( v52 || v31 != -8192 )
      v30 = v52;
    v29 = (v27 - 1) & (v51 + v29);
    v31 = *(_QWORD *)(v28 + 16LL * v29);
    if ( a2 == v31 )
      goto LABEL_12;
    ++v51;
    v52 = v30;
    v30 = (__int64 *)(v28 + 16LL * v29);
  }
  if ( !v52 )
    v52 = v30;
  v68 = v52;
  v53 = *(_DWORD *)(v21 + 104);
  ++*(_QWORD *)(v21 + 88);
  v54 = v53 + 1;
  if ( 4 * (v53 + 1) >= 3 * v27 )
    goto LABEL_63;
  v55 = a2;
  if ( v27 - *(_DWORD *)(v21 + 108) - v54 <= v27 >> 3 )
  {
    v67 = v14;
    goto LABEL_64;
  }
LABEL_54:
  *(_DWORD *)(v21 + 104) = v54;
  if ( *v52 != -4096 )
    --*(_DWORD *)(v21 + 108);
  *v52 = v55;
  v52[1] = (__int64)v70;
  v21 = a1[1];
LABEL_12:
  v69 = a2;
  v70 = v14;
  v32 = *(_DWORD *)(v21 + 144);
  if ( !v32 )
  {
    v68 = 0;
    ++*(_QWORD *)(v21 + 120);
LABEL_58:
    v32 *= 2;
    goto LABEL_59;
  }
  v33 = *(_QWORD *)(v21 + 128);
  v34 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = (__int64 *)(v33 + 16LL * v34);
  v36 = *result;
  if ( a2 == *result )
    return result;
  v43 = 1;
  v44 = 0;
  while ( v36 != -4096 )
  {
    if ( v44 || v36 != -8192 )
      result = v44;
    v34 = (v32 - 1) & (v43 + v34);
    v36 = *(_QWORD *)(v33 + 16LL * v34);
    if ( a2 == v36 )
      return result;
    ++v43;
    v44 = result;
    result = (__int64 *)(v33 + 16LL * v34);
  }
  if ( !v44 )
    v44 = result;
  v68 = v44;
  v45 = *(_DWORD *)(v21 + 136);
  ++*(_QWORD *)(v21 + 120);
  v46 = v45 + 1;
  if ( 4 * (v45 + 1) >= 3 * v32 )
    goto LABEL_58;
  if ( v32 - *(_DWORD *)(v21 + 140) - v46 <= v32 >> 3 )
  {
LABEL_59:
    sub_9BBF00(v21 + 120, v32);
    sub_28EE430(v21 + 120, &v69, &v68);
    a2 = v69;
    v44 = v68;
    v46 = *(_DWORD *)(v21 + 136) + 1;
  }
  *(_DWORD *)(v21 + 136) = v46;
  if ( *v44 != -4096 )
    --*(_DWORD *)(v21 + 140);
  *v44 = a2;
  result = v70;
  v44[1] = (__int64)v70;
  return result;
}
