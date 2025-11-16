// Function: sub_393AC30
// Address: 0x393ac30
//
__int64 *__fastcall sub_393AC30(__int64 *a1, __int64 a2, __m128i *a3)
{
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rdx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rdi
  _QWORD *v12; // rdi
  void (*v13)(void); // rax
  __int64 v14; // rdx
  unsigned __int16 *v15; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned __int64 *v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD **v22; // r12
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rdi
  const __m128i ****v25; // rax
  _QWORD *v26; // rcx
  _QWORD *v27; // r12
  _QWORD *v28; // rbx
  unsigned __int64 v29; // rdi
  _QWORD *v30; // r12
  _QWORD *v31; // rbx
  unsigned __int64 v32; // rdi
  const __m128i ***v33; // r12
  const __m128i ***v34; // rbx
  unsigned __int64 v35; // rdi
  _QWORD *v36; // r12
  _QWORD *v37; // rbx
  unsigned __int64 v38; // rdi
  unsigned __int64 *v39; // [rsp+0h] [rbp-70h]
  _QWORD *v40; // [rsp+0h] [rbp-70h]
  __int64 v41; // [rsp+8h] [rbp-68h]
  _QWORD *v42; // [rsp+8h] [rbp-68h]
  _QWORD **v43; // [rsp+8h] [rbp-68h]
  _QWORD *v44; // [rsp+8h] [rbp-68h]
  _QWORD *v45; // [rsp+8h] [rbp-68h]
  const __m128i ***v46; // [rsp+8h] [rbp-68h]
  _QWORD *v47; // [rsp+8h] [rbp-68h]
  unsigned __int64 v50; // [rsp+20h] [rbp-50h] BYREF
  const __m128i ****v51; // [rsp+28h] [rbp-48h] BYREF
  __int64 v52; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v53; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 32);
  v52 = 0;
  v53 = 0;
  (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64 *))(*(_QWORD *)v4 + 16LL))(&v50, v4, &v52);
  if ( (v50 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v5 = a2;
    v50 = v50 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_393AB10(a1, a2, (__int64 *)&v50);
    goto LABEL_3;
  }
  v8 = *(unsigned int *)(a2 + 48);
  v50 = 0;
  *(_DWORD *)(a2 + 48) = v8 + 1;
  v9 = v52 + 56 * v8;
  v5 = v9;
  sub_3937F60((__int64)a3, (char **)v9);
  v10 = *(_QWORD *)(v9 + 24);
  if ( !v10 )
  {
    v18 = (unsigned __int64 *)a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = 0;
    v39 = v18;
    if ( !v18 )
      goto LABEL_9;
    v19 = (_QWORD *)v18[3];
    v42 = (_QWORD *)v18[4];
    if ( v42 != v19 )
    {
      do
      {
        v20 = (_QWORD *)*v19;
        while ( v20 != v19 )
        {
          v21 = (unsigned __int64)v20;
          v20 = (_QWORD *)*v20;
          j_j___libc_free_0(v21);
        }
        v19 += 3;
      }
      while ( v42 != v19 );
      v19 = (_QWORD *)v39[3];
    }
    if ( v19 )
      j_j___libc_free_0((unsigned __int64)v19);
    v22 = (_QWORD **)*v39;
    v43 = (_QWORD **)v39[1];
    if ( v43 != (_QWORD **)*v39 )
    {
      do
      {
        v23 = *v22;
        while ( v23 != v22 )
        {
          v24 = (unsigned __int64)v23;
          v23 = (_QWORD *)*v23;
          j_j___libc_free_0(v24);
        }
        v22 += 3;
      }
      while ( v43 != v22 );
    }
    if ( *v39 )
      j_j___libc_free_0(*v39);
LABEL_30:
    v5 = 48;
    j_j___libc_free_0((unsigned __int64)v39);
    goto LABEL_9;
  }
  v11 = a3[1].m128i_i64[1];
  if ( !v11 )
  {
    v5 = *(_QWORD *)(v9 + 24);
    sub_39395F0(&v51, (const __m128i ***)v5);
    v6 = (__int64)a3;
    v25 = v51;
    v51 = 0;
    v26 = (_QWORD *)a3[1].m128i_i64[1];
    a3[1].m128i_i64[1] = (__int64)v25;
    v40 = v26;
    if ( !v26 )
      goto LABEL_9;
    v27 = (_QWORD *)v26[3];
    v44 = (_QWORD *)v26[4];
    if ( v44 != v27 )
    {
      do
      {
        v28 = (_QWORD *)*v27;
        while ( v28 != v27 )
        {
          v29 = (unsigned __int64)v28;
          v28 = (_QWORD *)*v28;
          j_j___libc_free_0(v29);
        }
        v27 += 3;
      }
      while ( v44 != v27 );
      v27 = (_QWORD *)v40[3];
    }
    if ( v27 )
      j_j___libc_free_0((unsigned __int64)v27);
    v30 = (_QWORD *)*v40;
    v45 = (_QWORD *)v40[1];
    if ( v45 != (_QWORD *)*v40 )
    {
      do
      {
        v31 = (_QWORD *)*v30;
        while ( v31 != v30 )
        {
          v32 = (unsigned __int64)v31;
          v31 = (_QWORD *)*v31;
          j_j___libc_free_0(v32);
        }
        v30 += 3;
      }
      while ( v45 != v30 );
      v30 = (_QWORD *)*v40;
    }
    if ( v30 )
      j_j___libc_free_0((unsigned __int64)v30);
    v5 = 48;
    j_j___libc_free_0((unsigned __int64)v40);
    v39 = (unsigned __int64 *)v51;
    if ( !v51 )
      goto LABEL_9;
    v33 = v51[3];
    v46 = v51[4];
    if ( v46 != v33 )
    {
      do
      {
        v34 = (const __m128i ***)*v33;
        if ( *v33 != (const __m128i **)v33 )
        {
          do
          {
            v35 = (unsigned __int64)v34;
            v34 = (const __m128i ***)*v34;
            j_j___libc_free_0(v35);
          }
          while ( v34 != v33 );
        }
        v33 += 3;
      }
      while ( v46 != v33 );
      v33 = (const __m128i ***)v39[3];
    }
    if ( v33 )
      j_j___libc_free_0((unsigned __int64)v33);
    v36 = (_QWORD *)*v39;
    v47 = (_QWORD *)v39[1];
    if ( v47 != (_QWORD *)*v39 )
    {
      do
      {
        v37 = (_QWORD *)*v36;
        while ( v37 != v36 )
        {
          v38 = (unsigned __int64)v37;
          v37 = (_QWORD *)*v37;
          j_j___libc_free_0(v38);
        }
        v36 += 3;
      }
      while ( v47 != v36 );
      v36 = (_QWORD *)*v39;
    }
    if ( v36 )
      j_j___libc_free_0((unsigned __int64)v36);
    goto LABEL_30;
  }
  v41 = a3[1].m128i_i64[1];
  sub_3938290(v11, *(const __m128i ****)(v9 + 24));
  v5 = v10 + 24;
  sub_3938290(v41 + 24, (const __m128i ***)(v10 + 24));
LABEL_9:
  a3[2] = _mm_loadu_si128((const __m128i *)(v9 + 32));
  a3[3].m128i_i64[0] = *(_QWORD *)(v9 + 48);
  if ( *(unsigned int *)(a2 + 48) >= v53 )
  {
    v12 = *(_QWORD **)(a2 + 32);
    v13 = *(void (**)(void))(*v12 + 32LL);
    if ( (char *)v13 == (char *)sub_3937F10 )
    {
      v14 = v12[3];
      v15 = (unsigned __int16 *)v12[2];
      if ( !v14 )
      {
        v14 = *v15++;
        v12[3] = v14;
      }
      v5 = (__int64)(v15 + 8);
      v6 = v14 - 1;
      v12[2] = v15 + 4;
      v16 = *((_QWORD *)v15 + 1);
      v12[2] = v15 + 8;
      v17 = (__int64)v15 + *((_QWORD *)v15 + 2) + v16 + 24;
      --v12[4];
      v12[2] = v17;
      v12[3] = v6;
    }
    else
    {
      v13();
    }
    *(_DWORD *)(a2 + 48) = 0;
  }
  *(_DWORD *)(a2 + 8) = 0;
  *a1 = 1;
LABEL_3:
  if ( (v50 & 1) != 0 || (v50 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v50, v5, v6);
  return a1;
}
