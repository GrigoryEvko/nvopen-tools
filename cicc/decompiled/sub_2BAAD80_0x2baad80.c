// Function: sub_2BAAD80
// Address: 0x2baad80
//
void __fastcall sub_2BAAD80(__int64 *a1, __int64 *a2, unsigned __int64 a3, int a4)
{
  __m128i v8; // rax
  __int64 v9; // r9
  _DWORD *v10; // r8
  __int64 v11; // rdx
  int v12; // edi
  __int64 v13; // r10
  int v14; // edi
  __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rcx
  unsigned __int64 v19; // rbx
  _BYTE *v20; // r10
  __int64 v21; // rdi
  __int64 v22; // rax
  bool v23; // al
  __int64 v24; // rcx
  __int64 v25; // rbx
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rdi
  unsigned int v28; // esi
  int v29; // eax
  unsigned int *v30; // rdx
  __int64 v31; // rdi
  unsigned __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rbx
  __int64 v35; // rsi
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdx
  int v38; // eax
  _DWORD *v39; // rdx
  __int64 v40; // rdi
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  int v44; // esi
  int v45; // r11d
  _DWORD *v46; // [rsp+10h] [rbp-A0h]
  _DWORD *v47; // [rsp+10h] [rbp-A0h]
  _DWORD *v48; // [rsp+10h] [rbp-A0h]
  _BYTE *v49; // [rsp+18h] [rbp-98h]
  _DWORD *v50; // [rsp+18h] [rbp-98h]
  __int64 v51; // [rsp+18h] [rbp-98h]
  __int64 v52; // [rsp+18h] [rbp-98h]
  __m128i v53; // [rsp+20h] [rbp-90h] BYREF
  __int64 v54; // [rsp+30h] [rbp-80h]
  _BYTE *v55; // [rsp+40h] [rbp-70h] BYREF
  __int64 v56; // [rsp+48h] [rbp-68h]
  _BYTE v57[96]; // [rsp+50h] [rbp-60h] BYREF

  v8.m128i_i64[0] = sub_2B5F980(a2, a3, *(__int64 **)(*a1 + 3304));
  v53 = v8;
  if ( !v8.m128i_i64[0] || !v53.m128i_i64[1] )
  {
    v11 = *a1;
    v10 = &v55;
    goto LABEL_26;
  }
  v10 = &v55;
  if ( *(_BYTE *)v8.m128i_i64[0] == 61 )
    goto LABEL_18;
  v11 = *a1;
  if ( (*(_BYTE *)(*a1 + 88) & 1) != 0 )
  {
    v13 = v11 + 96;
    v14 = 3;
  }
  else
  {
    v12 = *(_DWORD *)(v11 + 104);
    v13 = *(_QWORD *)(v11 + 96);
    if ( !v12 )
      goto LABEL_26;
    v14 = v12 - 1;
  }
  v15 = v14 & (((unsigned __int32)v8.m128i_i32[0] >> 9) ^ ((unsigned __int32)v8.m128i_i32[0] >> 4));
  v16 = v13 + 72 * v15;
  v17 = *(_QWORD *)v16;
  if ( v8.m128i_i64[0] != *(_QWORD *)v16 )
  {
    v44 = 1;
    while ( v17 != -4096 )
    {
      v45 = v44 + 1;
      v15 = v14 & (unsigned int)(v44 + v15);
      v16 = v13 + 72LL * (unsigned int)v15;
      v17 = *(_QWORD *)v16;
      if ( v8.m128i_i64[0] == *(_QWORD *)v16 )
        goto LABEL_8;
      v44 = v45;
    }
    v10 = &v55;
    goto LABEL_26;
  }
LABEL_8:
  v10 = &v55;
  v55 = v57;
  v56 = 0x600000000LL;
  if ( !*(_DWORD *)(v16 + 16) )
  {
LABEL_26:
    v33 = 0;
    v34 = *(_QWORD *)a1[1];
    if ( a4 )
      v33 = *(unsigned int *)(a1[2] + 8);
    v35 = *(unsigned int *)(v11 + 8);
    v36 = *(unsigned int *)(v34 + 220);
    v37 = *(unsigned int *)(v34 + 216);
    v38 = *(_DWORD *)(v34 + 216);
    if ( v37 >= v36 )
    {
      v42 = (v33 << 32) | v35;
      if ( v36 < v37 + 1 )
      {
        v47 = v10;
        v51 = (v33 << 32) | v35;
        sub_C8D5F0(v34 + 208, (const void *)(v34 + 224), v37 + 1, 8u, (__int64)v10, v9);
        v37 = *(unsigned int *)(v34 + 216);
        v10 = v47;
        v42 = v51;
      }
      *(_QWORD *)(*(_QWORD *)(v34 + 208) + 8 * v37) = v42;
      ++*(_DWORD *)(v34 + 216);
    }
    else
    {
      v39 = (_DWORD *)(*(_QWORD *)(v34 + 208) + 8 * v37);
      if ( v39 )
      {
        *v39 = v35;
        v39[1] = v33;
        v38 = *(_DWORD *)(v34 + 216);
      }
      *(_DWORD *)(v34 + 216) = v38 + 1;
    }
    v40 = *a1;
    v41 = *(_QWORD *)a1[1];
    LODWORD(v56) = a4;
    v55 = (_BYTE *)v41;
    sub_2BA65A0(v40, a2, a3, *(_DWORD *)a1[3], v10, 0);
    return;
  }
  sub_2B0C870((__int64)&v55, v16 + 8, v11, v15, (__int64)&v55, v9);
  v19 = (unsigned __int64)v55;
  v10 = &v55;
  v20 = &v55[8 * (unsigned int)v56];
  if ( v55 == v20 )
  {
LABEL_35:
    if ( v20 == v57 )
    {
      v11 = *a1;
    }
    else
    {
      v50 = v10;
      _libc_free((unsigned __int64)v20);
      v11 = *a1;
      v10 = v50;
    }
    goto LABEL_26;
  }
  while ( 1 )
  {
    v21 = *(_QWORD *)v19;
    v22 = *(unsigned int *)(*(_QWORD *)v19 + 120LL);
    if ( !(_DWORD)v22 )
      v22 = *(unsigned int *)(v21 + 8);
    if ( v22 == a3 )
    {
      v46 = v10;
      v49 = v20;
      v23 = sub_2B31C30(v21, (char *)a2, a3, v18, (__int64)v10, v9);
      v20 = v49;
      v10 = v46;
      if ( v23 )
        break;
    }
    v19 += 8LL;
    if ( v20 == (_BYTE *)v19 )
    {
      v20 = v55;
      goto LABEL_35;
    }
  }
  if ( v55 != v57 )
  {
    _libc_free((unsigned __int64)v55);
    v10 = v46;
  }
LABEL_18:
  v24 = 0;
  v25 = *(_QWORD *)a1[1];
  if ( a4 )
    v24 = *(unsigned int *)(a1[2] + 8);
  v26 = *(unsigned int *)(v25 + 216);
  v27 = *(unsigned int *)(v25 + 220);
  v28 = *(_DWORD *)(*a1 + 8);
  v29 = *(_DWORD *)(v25 + 216);
  if ( v26 >= v27 )
  {
    v43 = (v24 << 32) | v28;
    if ( v27 < v26 + 1 )
    {
      v48 = v10;
      v52 = (v24 << 32) | v28;
      sub_C8D5F0(v25 + 208, (const void *)(v25 + 224), v26 + 1, 8u, (__int64)v10, v9);
      v26 = *(unsigned int *)(v25 + 216);
      v10 = v48;
      v43 = v52;
    }
    *(_QWORD *)(*(_QWORD *)(v25 + 208) + 8 * v26) = v43;
    ++*(_DWORD *)(v25 + 216);
  }
  else
  {
    v30 = (unsigned int *)(*(_QWORD *)(v25 + 208) + 8 * v26);
    if ( v30 )
    {
      *v30 = v28;
      v30[1] = v24;
      v29 = *(_DWORD *)(v25 + 216);
    }
    *(_DWORD *)(v25 + 216) = v29 + 1;
  }
  v31 = *a1;
  v32 = *(_QWORD *)a1[1];
  LODWORD(v56) = a4;
  v55 = (_BYTE *)v32;
  sub_2B70420(v31, a2, a3, 3, v54, &v53, v10, 0, 0, 0, 0);
}
