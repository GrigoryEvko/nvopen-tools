// Function: sub_393B300
// Address: 0x393b300
//
unsigned __int64 __fastcall sub_393B300(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned int *a4,
        unsigned __int64 a5)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v7; // r14
  unsigned __int64 v8; // r13
  _QWORD *v9; // r12
  _QWORD *v10; // r15
  unsigned __int64 v11; // rdi
  _QWORD *v12; // r12
  _QWORD *v13; // r15
  unsigned __int64 v14; // rdi
  unsigned int *v15; // rax
  unsigned int *v16; // r12
  _BYTE *v17; // rdi
  unsigned __int64 *v18; // rdx
  unsigned __int64 v19; // rcx
  bool v20; // zf
  unsigned int *v21; // rax
  unsigned __int64 v22; // r15
  __int64 v23; // r14
  _BYTE *v24; // rsi
  __int64 v25; // rdx
  __m128i *v26; // rsi
  unsigned __int64 v27; // rdi
  _BYTE *v28; // r9
  _BYTE *v29; // rax
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // r8
  __int64 v32; // rcx
  unsigned __int64 *v33; // r14
  unsigned __int64 v34; // r13
  _QWORD *v35; // r12
  _QWORD *v36; // r15
  unsigned __int64 v37; // rdi
  _QWORD *v38; // r12
  _QWORD *v39; // r15
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // [rsp+8h] [rbp-98h]
  unsigned __int64 v43; // [rsp+10h] [rbp-90h]
  unsigned __int64 *v44; // [rsp+18h] [rbp-88h]
  unsigned __int64 *v45; // [rsp+18h] [rbp-88h]
  _QWORD *v46; // [rsp+20h] [rbp-80h]
  _QWORD *v47; // [rsp+20h] [rbp-80h]
  _QWORD *v48; // [rsp+20h] [rbp-80h]
  _QWORD *v49; // [rsp+20h] [rbp-80h]
  unsigned __int64 v50; // [rsp+20h] [rbp-80h]
  unsigned int *v51; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v52; // [rsp+30h] [rbp-70h] BYREF
  __int64 v53; // [rsp+38h] [rbp-68h]
  unsigned __int64 v54; // [rsp+40h] [rbp-60h] BYREF
  __int64 v55; // [rsp+48h] [rbp-58h] BYREF
  _BYTE *v56; // [rsp+50h] [rbp-50h] BYREF
  _BYTE *v57; // [rsp+58h] [rbp-48h]
  _BYTE *v58; // [rsp+60h] [rbp-40h]

  v52 = a2;
  v53 = a3;
  v51 = a4;
  if ( (a5 & 7) != 0 )
    return 0;
  v41 = *(_QWORD *)a1;
  v44 = *(unsigned __int64 **)(a1 + 8);
  if ( *(unsigned __int64 **)a1 != v44 )
  {
    v7 = *(unsigned __int64 **)a1;
    do
    {
      v8 = v7[3];
      if ( v8 )
      {
        v9 = *(_QWORD **)(v8 + 24);
        v46 = *(_QWORD **)(v8 + 32);
        if ( v46 != v9 )
        {
          do
          {
            v10 = (_QWORD *)*v9;
            while ( v10 != v9 )
            {
              v11 = (unsigned __int64)v10;
              v10 = (_QWORD *)*v10;
              j_j___libc_free_0(v11);
            }
            v9 += 3;
          }
          while ( v46 != v9 );
          v9 = *(_QWORD **)(v8 + 24);
        }
        if ( v9 )
          j_j___libc_free_0((unsigned __int64)v9);
        v12 = *(_QWORD **)v8;
        v47 = *(_QWORD **)(v8 + 8);
        if ( v47 != *(_QWORD **)v8 )
        {
          do
          {
            v13 = (_QWORD *)*v12;
            while ( v13 != v12 )
            {
              v14 = (unsigned __int64)v13;
              v13 = (_QWORD *)*v13;
              j_j___libc_free_0(v14);
            }
            v12 += 3;
          }
          while ( v47 != v12 );
          v12 = *(_QWORD **)v8;
        }
        if ( v12 )
          j_j___libc_free_0((unsigned __int64)v12);
        j_j___libc_free_0(v8);
      }
      if ( *v7 )
        j_j___libc_free_0(*v7);
      v7 += 7;
    }
    while ( v44 != v7 );
    *(_QWORD *)(a1 + 8) = v41;
  }
  v15 = v51;
  v56 = 0;
  v57 = 0;
  v16 = (unsigned int *)((char *)v51 + a5);
  v58 = 0;
  if ( v51 >= (unsigned int *)((char *)v51 + a5) )
  {
    v17 = 0;
LABEL_48:
    result = *(_QWORD *)a1;
  }
  else
  {
    v17 = 0;
    while ( 1 )
    {
      v18 = (unsigned __int64 *)(v15 + 2);
      if ( v15 + 2 >= v16 )
        goto LABEL_77;
      v19 = *(_QWORD *)v15;
      v20 = *(_DWORD *)(a1 + 28) == 1;
      v51 = v15 + 2;
      v54 = v19;
      if ( v20 )
      {
        v22 = (a5 >> 3) - 1;
      }
      else
      {
        v21 = v15 + 4;
        if ( v21 > v16 )
          goto LABEL_77;
        v22 = *v18;
        v51 = v21;
        v18 = (unsigned __int64 *)v21;
      }
      if ( v16 < (unsigned int *)&v18[v22] )
      {
LABEL_77:
        result = 0;
        goto LABEL_78;
      }
      if ( v57 != v17 )
        v57 = v17;
      v23 = 0;
      sub_9C9810((__int64)&v56, v22);
      if ( v22 )
      {
        do
        {
          while ( 1 )
          {
            v24 = v57;
            v25 = *(_QWORD *)v51;
            v51 += 2;
            v55 = v25;
            if ( v57 != v58 )
              break;
            ++v23;
            sub_A235E0((__int64)&v56, v57, &v55);
            if ( v22 == v23 )
              goto LABEL_42;
          }
          if ( v57 )
          {
            *(_QWORD *)v57 = v25;
            v24 = v57;
          }
          ++v23;
          v57 = v24 + 8;
        }
        while ( v22 != v23 );
      }
LABEL_42:
      v26 = *(__m128i **)(a1 + 8);
      if ( v26 == *(__m128i **)(a1 + 16) )
      {
        sub_3939AB0((unsigned __int64 *)a1, v26, &v52, &v54, (unsigned __int64 *)&v56);
      }
      else
      {
        v27 = (unsigned __int64)v56;
        v28 = v57;
        v56 = 0;
        v29 = v58;
        v30 = v54;
        v58 = 0;
        v31 = v52;
        v32 = v53;
        v57 = 0;
        if ( v26 )
        {
          v26->m128i_i64[0] = v27;
          v26->m128i_i64[1] = (__int64)v28;
          v26[1].m128i_i64[0] = (__int64)v29;
          v26[1].m128i_i64[1] = 0;
          v26[2].m128i_i64[0] = v31;
          v26[2].m128i_i64[1] = v32;
          v26[3].m128i_i64[0] = v30;
        }
        else if ( v27 )
        {
          j_j___libc_free_0(v27);
        }
        *(_QWORD *)(a1 + 8) += 56LL;
      }
      if ( *(_DWORD *)(a1 + 28) > 2u && !(unsigned __int8)sub_393B230(a1, &v51, (unsigned __int64)v16) )
        break;
      v15 = v51;
      v17 = v56;
      if ( v51 >= v16 )
        goto LABEL_48;
    }
    v43 = *(_QWORD *)a1;
    v45 = *(unsigned __int64 **)(a1 + 8);
    if ( *(unsigned __int64 **)a1 != v45 )
    {
      v33 = *(unsigned __int64 **)a1;
      do
      {
        v34 = v33[3];
        if ( v34 )
        {
          v35 = *(_QWORD **)(v34 + 24);
          v48 = *(_QWORD **)(v34 + 32);
          if ( v48 != v35 )
          {
            do
            {
              v36 = (_QWORD *)*v35;
              while ( v35 != v36 )
              {
                v37 = (unsigned __int64)v36;
                v36 = (_QWORD *)*v36;
                j_j___libc_free_0(v37);
              }
              v35 += 3;
            }
            while ( v48 != v35 );
            v35 = *(_QWORD **)(v34 + 24);
          }
          if ( v35 )
            j_j___libc_free_0((unsigned __int64)v35);
          v38 = *(_QWORD **)v34;
          v49 = *(_QWORD **)(v34 + 8);
          if ( v49 != *(_QWORD **)v34 )
          {
            do
            {
              v39 = (_QWORD *)*v38;
              while ( v39 != v38 )
              {
                v40 = (unsigned __int64)v39;
                v39 = (_QWORD *)*v39;
                j_j___libc_free_0(v40);
              }
              v38 += 3;
            }
            while ( v49 != v38 );
            v38 = *(_QWORD **)v34;
          }
          if ( v38 )
            j_j___libc_free_0((unsigned __int64)v38);
          j_j___libc_free_0(v34);
        }
        if ( *v33 )
          j_j___libc_free_0(*v33);
        v33 += 7;
      }
      while ( v45 != v33 );
      *(_QWORD *)(a1 + 8) = v43;
    }
    v17 = v56;
    result = 0;
  }
LABEL_78:
  if ( v17 )
  {
    v50 = result;
    j_j___libc_free_0((unsigned __int64)v17);
    return v50;
  }
  return result;
}
