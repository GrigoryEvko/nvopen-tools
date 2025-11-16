// Function: sub_30BE5D0
// Address: 0x30be5d0
//
void __fastcall sub_30BE5D0(__int64 *a1)
{
  __int64 (*v1)(void); // rax
  __int64 v2; // rax
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 *v9; // rdi
  __m128i *v10; // rsi
  const __m128i *v11; // rdx
  const __m128i *v12; // rax
  unsigned __int64 v13; // rbx
  __int64 v14; // rax
  __m128i *v15; // r15
  __m128i *v16; // rax
  __m128i *v17; // rax
  const __m128i *v18; // rdx
  unsigned __int64 v19; // r12
  signed __int64 v20; // rbx
  __int64 v21; // r13
  __int64 v22; // rbx
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // rdi
  __int64 v28; // rbx
  __int64 *v29; // rax
  char v30; // dl
  unsigned __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 (__fastcall *v33)(__int64, __int64, __int64); // rax
  __int64 v34; // rax
  __int64 *v35; // [rsp+10h] [rbp-130h]
  __int64 v37; // [rsp+20h] [rbp-120h]
  __int64 v38; // [rsp+28h] [rbp-118h]
  __int64 *v39; // [rsp+30h] [rbp-110h]
  __int64 v40; // [rsp+38h] [rbp-108h]
  __m128i v41; // [rsp+48h] [rbp-F8h] BYREF
  __int64 *v42; // [rsp+58h] [rbp-E8h] BYREF
  __int64 *v43; // [rsp+60h] [rbp-E0h]
  char *v44; // [rsp+68h] [rbp-D8h]
  __m128i v45; // [rsp+70h] [rbp-D0h] BYREF
  char v46; // [rsp+88h] [rbp-B8h]
  __int64 v47; // [rsp+90h] [rbp-B0h] BYREF
  char *v48; // [rsp+98h] [rbp-A8h]
  __int64 v49; // [rsp+A0h] [rbp-A0h]
  int v50; // [rsp+A8h] [rbp-98h]
  char v51; // [rsp+ACh] [rbp-94h]
  char v52; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v53; // [rsp+D0h] [rbp-70h] BYREF
  const __m128i *v54; // [rsp+D8h] [rbp-68h]
  const __m128i *v55; // [rsp+E0h] [rbp-60h]
  const __m128i *v56; // [rsp+F8h] [rbp-48h]
  const __m128i *v57; // [rsp+100h] [rbp-40h]

  v1 = *(__int64 (**)(void))(*a1 + 16);
  if ( (char *)v1 == (char *)sub_30B2870 )
  {
    v2 = sub_22077B0(0x40u);
    if ( v2 )
    {
      *(_QWORD *)(v2 + 8) = 0;
      *(_QWORD *)(v2 + 40) = v2 + 56;
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v2 + 24) = 0;
      *(_DWORD *)(v2 + 32) = 0;
      *(_QWORD *)(v2 + 48) = 0;
      *(_DWORD *)(v2 + 56) = 4;
      *(_QWORD *)v2 = &unk_4A32368;
    }
    v38 = v2;
    sub_30B2450(a1[1], v2);
  }
  else
  {
    v38 = v1();
  }
  v5 = a1[1];
  v47 = 0;
  v48 = &v52;
  v6 = *(__int64 **)(v5 + 96);
  v7 = *(unsigned int *)(v5 + 104);
  v49 = 4;
  v37 = v38 + 8;
  v8 = (__int64)&v53;
  v51 = 1;
  v50 = 0;
  v35 = &v6[v7];
  v39 = v6;
  if ( v35 == v6 )
    return;
  while ( 2 )
  {
    v41.m128i_i64[0] = *v39;
    if ( v37 == v41.m128i_i64[0] + 8 )
      goto LABEL_7;
    v9 = &v53;
    v10 = &v41;
    sub_30BA8B0(&v53, &v41, (__int64)&v47, v8, v3, v4);
    v11 = v54;
    v42 = 0;
    v43 = 0;
    v41.m128i_i64[1] = v53;
    v12 = v55;
    v44 = 0;
    v13 = (char *)v55 - (char *)v54;
    if ( v55 == v54 )
    {
      v9 = 0;
    }
    else
    {
      if ( v13 > 0x7FFFFFFFFFFFFFE0LL )
        goto LABEL_70;
      v14 = sub_22077B0((char *)v55 - (char *)v54);
      v11 = v54;
      v9 = (__int64 *)v14;
      v12 = v55;
    }
    v42 = v9;
    v43 = v9;
    v44 = (char *)v9 + v13;
    if ( v11 == v12 )
    {
      v15 = (__m128i *)v9;
    }
    else
    {
      v15 = (__m128i *)((char *)v9 + (char *)v12 - (char *)v11);
      v16 = (__m128i *)v9;
      do
      {
        if ( v16 )
        {
          *v16 = _mm_loadu_si128(v11);
          v16[1] = _mm_loadu_si128(v11 + 1);
        }
        v16 += 2;
        v11 += 2;
      }
      while ( v16 != v15 );
    }
    v11 = v56;
    v43 = (__int64 *)v15;
    v10 = (__m128i *)((char *)v57 - (char *)v56);
    if ( v57 == v56 )
    {
      v19 = 0;
      goto LABEL_63;
    }
    if ( (unsigned __int64)v10 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_70:
      sub_4261EA(v9, v10, v11);
    v17 = (__m128i *)sub_22077B0((char *)v57 - (char *)v56);
    v18 = v56;
    v9 = v42;
    v19 = (unsigned __int64)v17;
    v15 = (__m128i *)v43;
    if ( v56 != v57 )
    {
      v20 = (char *)v57 - (char *)v56;
      v8 = (__int64)v17->m128i_i64 + (char *)v57 - (char *)v56;
      do
      {
        if ( v17 )
        {
          *v17 = _mm_loadu_si128(v18);
          v17[1] = _mm_loadu_si128(v18 + 1);
        }
        v17 += 2;
        v18 += 2;
      }
      while ( (__m128i *)v8 != v17 );
      v40 = v20;
      goto LABEL_25;
    }
LABEL_63:
    v40 = 0;
LABEL_25:
    if ( v40 == (char *)v15 - (char *)v9 )
      goto LABEL_38;
    while ( 2 )
    {
      while ( 2 )
      {
        v21 = v15[-2].m128i_i64[0];
        v22 = v41.m128i_i64[0];
        if ( v41.m128i_i64[0] == v21 )
        {
          v33 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*a1 + 56);
          if ( v33 == sub_30B2F90 )
          {
            v34 = sub_22077B0(0x10u);
            if ( v34 )
            {
              *(_QWORD *)v34 = v22;
              *(_DWORD *)(v34 + 8) = 3;
            }
            v45.m128i_i64[0] = v34;
            sub_30B2AC0(v37, v45.m128i_i64);
            v15 = (__m128i *)v43;
            v21 = *(v43 - 4);
          }
          else
          {
            v33((__int64)a1, v38, v15[-2].m128i_i64[0]);
            v15 = (__m128i *)v43;
LABEL_56:
            v21 = v15[-2].m128i_i64[0];
          }
        }
        if ( !v15[-1].m128i_i8[8] )
        {
          v23 = *(_QWORD **)(v21 + 40);
          v15[-1].m128i_i8[8] = 1;
          v15[-2].m128i_i64[1] = (__int64)v23;
          v15[-1].m128i_i64[0] = (__int64)sub_30B9540;
          goto LABEL_29;
        }
        while ( 1 )
        {
          while ( 1 )
          {
            v23 = (_QWORD *)v15[-2].m128i_i64[1];
LABEL_29:
            v8 = *(unsigned int *)(v21 + 48);
            if ( v23 == (_QWORD *)(*(_QWORD *)(v21 + 40) + 8 * v8) )
            {
              v43 -= 4;
              v9 = v42;
              v15 = (__m128i *)v43;
              if ( v43 == v42 )
                goto LABEL_25;
              goto LABEL_56;
            }
            v15[-2].m128i_i64[1] = (__int64)(v23 + 1);
            v24 = ((__int64 (__fastcall *)(_QWORD))v15[-1].m128i_i64[0])(*v23);
            v27 = (_QWORD *)v41.m128i_i64[1];
            v28 = v24;
            if ( *(_BYTE *)(v41.m128i_i64[1] + 28) )
              break;
LABEL_36:
            sub_C8CC70(v41.m128i_i64[1], v28, (__int64)v25, v26, v3, v4);
            if ( v30 )
              goto LABEL_37;
          }
          v29 = *(__int64 **)(v41.m128i_i64[1] + 8);
          v26 = *(unsigned int *)(v41.m128i_i64[1] + 20);
          v25 = &v29[v26];
          if ( v29 == v25 )
            break;
          while ( v28 != *v29 )
          {
            if ( v25 == ++v29 )
              goto LABEL_53;
          }
        }
LABEL_53:
        if ( (unsigned int)v26 >= *(_DWORD *)(v41.m128i_i64[1] + 16) )
          goto LABEL_36;
        *(_DWORD *)(v41.m128i_i64[1] + 20) = v26 + 1;
        *v25 = v28;
        ++*v27;
LABEL_37:
        v45.m128i_i64[0] = v28;
        v46 = 0;
        sub_30BA870((unsigned __int64 *)&v42, &v45);
        v15 = (__m128i *)v43;
        v9 = v42;
        if ( v40 != (char *)v43 - (char *)v42 )
          continue;
        break;
      }
LABEL_38:
      if ( v9 != (__int64 *)v15 )
      {
        v31 = v19;
        v32 = v9;
        while ( *v32 == *(_QWORD *)v31 )
        {
          v8 = *((unsigned __int8 *)v32 + 24);
          if ( (_BYTE)v8 != *(_BYTE *)(v31 + 24) || (_BYTE)v8 && v32[1] != *(_QWORD *)(v31 + 8) )
            break;
          v32 += 4;
          v31 += 32LL;
          if ( v32 == (__int64 *)v15 )
            goto LABEL_45;
        }
        continue;
      }
      break;
    }
LABEL_45:
    if ( v19 )
    {
      j_j___libc_free_0(v19);
      v9 = v42;
    }
    if ( v9 )
      j_j___libc_free_0((unsigned __int64)v9);
    if ( v56 )
      j_j___libc_free_0((unsigned __int64)v56);
    if ( v54 )
      j_j___libc_free_0((unsigned __int64)v54);
LABEL_7:
    if ( v35 != ++v39 )
      continue;
    break;
  }
  if ( !v51 )
    _libc_free((unsigned __int64)v48);
}
