// Function: sub_2F09BF0
// Address: 0x2f09bf0
//
void __fastcall sub_2F09BF0(unsigned __int64 *a1, unsigned __int64 *a2, char *a3)
{
  unsigned __int64 **v3; // r14
  unsigned __int64 *v4; // r8
  unsigned __int64 v5; // rbx
  signed __int64 v6; // rcx
  _QWORD *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 **v10; // r15
  __int64 v11; // r13
  __int64 v12; // rdx
  unsigned __int64 **v13; // rsi
  __int64 v14; // rdi
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // r15
  unsigned __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 *v19; // r15
  unsigned __int64 *i; // r12
  unsigned __int64 v21; // rbx
  __m128i *v22; // r14
  unsigned __int64 v23; // rbx
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // r14
  unsigned __int64 v26; // r12
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r15
  __int64 v29; // r15
  unsigned __int64 *v30; // rbx
  unsigned __int64 *v31; // r12
  unsigned __int64 v32; // rax
  char *v33; // r14
  unsigned __int64 v34; // rbx
  __m128i *v35; // r15
  _BYTE **v36; // rbx
  _BYTE **v37; // r13
  unsigned __int64 *v38; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v39; // [rsp+10h] [rbp-50h]
  signed __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v41; // [rsp+20h] [rbp-40h]
  unsigned __int64 *v42; // [rsp+28h] [rbp-38h]

  v41 = a1;
  if ( a2 != a1 )
  {
    v3 = (unsigned __int64 **)a2;
    v4 = (unsigned __int64 *)*a2;
    v5 = *a1;
    v42 = (unsigned __int64 *)a2[1];
    v6 = (signed __int64)v42 - *a2;
    v40 = v6;
    if ( a1[2] - *a1 < v6 )
    {
      if ( v6 )
      {
        if ( (unsigned __int64)v6 > 0x7FFFFFFFFFFFFFE0LL )
LABEL_67:
          sub_4261EA(a1, a2, a3);
        a1 = (unsigned __int64 *)((char *)v42 - *a2);
        v38 = (unsigned __int64 *)*a2;
        v18 = sub_22077B0(v6);
        v4 = v38;
        v39 = (unsigned __int64 *)v18;
      }
      else
      {
        v39 = 0;
      }
      v19 = v39;
      for ( i = v4; v42 != i; i += 4 )
      {
        if ( v19 )
        {
          *v19 = *i;
          v21 = i[2] - i[1];
          v19[1] = 0;
          v19[2] = 0;
          v19[3] = 0;
          if ( v21 )
          {
            if ( v21 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_67;
            a1 = (unsigned __int64 *)v21;
            v22 = (__m128i *)sub_22077B0(v21);
          }
          else
          {
            v21 = 0;
            v22 = 0;
          }
          v19[1] = (unsigned __int64)v22;
          v19[2] = (unsigned __int64)v22;
          v19[3] = (unsigned __int64)v22->m128i_u64 + v21;
          v23 = i[2];
          if ( v23 != i[1] )
          {
            v24 = i[1];
            do
            {
              if ( v22 )
              {
                a1 = (unsigned __int64 *)v22;
                v22->m128i_i64[0] = (__int64)v22[1].m128i_i64;
                a2 = *(unsigned __int64 **)v24;
                sub_2F07250(v22->m128i_i64, *(_BYTE **)v24, *(_QWORD *)v24 + *(_QWORD *)(v24 + 8));
                v22[2] = _mm_loadu_si128((const __m128i *)(v24 + 32));
                a3 = (char *)*(unsigned __int16 *)(v24 + 48);
                v22[3].m128i_i16[0] = (__int16)a3;
              }
              v24 += 56LL;
              v22 = (__m128i *)((char *)v22 + 56);
            }
            while ( v23 != v24 );
          }
          v19[2] = (unsigned __int64)v22;
        }
        v19 += 4;
      }
      v25 = v41[1];
      v26 = *v41;
      if ( v25 != *v41 )
      {
        do
        {
          v27 = *(unsigned __int64 **)(v26 + 16);
          v28 = *(unsigned __int64 **)(v26 + 8);
          if ( v27 != v28 )
          {
            do
            {
              if ( (unsigned __int64 *)*v28 != v28 + 2 )
                j_j___libc_free_0(*v28);
              v28 += 7;
            }
            while ( v27 != v28 );
            v28 = *(unsigned __int64 **)(v26 + 8);
          }
          if ( v28 )
            j_j___libc_free_0((unsigned __int64)v28);
          v26 += 32LL;
        }
        while ( v25 != v26 );
        v26 = *v41;
      }
      if ( v26 )
        j_j___libc_free_0(v26);
      v17 = (unsigned __int64)v39 + v40;
      *v41 = (unsigned __int64)v39;
      v41[2] = (unsigned __int64)v39 + v40;
      goto LABEL_17;
    }
    v7 = (_QWORD *)a1[1];
    v8 = (__int64)v7 - v5;
    a3 = (char *)v7 - v5;
    if ( v6 > (unsigned __int64)v7 - v5 )
    {
      v29 = v8 >> 5;
      if ( v8 > 0 )
      {
        v30 = (unsigned __int64 *)(v5 + 8);
        v31 = v4 + 1;
        do
        {
          v32 = *(v31 - 1);
          a2 = v31;
          a1 = v30;
          v31 += 4;
          v30 += 4;
          *(v30 - 5) = v32;
          sub_2F092C0((__int64)a1, (unsigned __int64 **)a2);
          --v29;
        }
        while ( v29 );
        v4 = *v3;
        v7 = (_QWORD *)v41[1];
        v5 = *v41;
        v42 = v3[1];
        a3 = (char *)v7 - *v41;
      }
      v33 = &a3[(_QWORD)v4];
      v17 = v40 + v5;
      if ( (unsigned __int64 *)&a3[(_QWORD)v4] == v42 )
        goto LABEL_17;
      do
      {
        if ( v7 )
        {
          *v7 = *(_QWORD *)v33;
          v34 = *((_QWORD *)v33 + 2) - *((_QWORD *)v33 + 1);
          v7[1] = 0;
          v7[2] = 0;
          v7[3] = 0;
          if ( v34 )
          {
            if ( v34 > 0x7FFFFFFFFFFFFFF8LL )
              goto LABEL_67;
            a1 = (unsigned __int64 *)v34;
            v35 = (__m128i *)sub_22077B0(v34);
          }
          else
          {
            v35 = 0;
          }
          v7[1] = v35;
          v7[2] = v35;
          v7[3] = (char *)v35 + v34;
          v36 = (_BYTE **)*((_QWORD *)v33 + 2);
          if ( v36 != *((_BYTE ***)v33 + 1) )
          {
            v37 = (_BYTE **)*((_QWORD *)v33 + 1);
            do
            {
              if ( v35 )
              {
                a1 = (unsigned __int64 *)v35;
                v35->m128i_i64[0] = (__int64)v35[1].m128i_i64;
                a2 = (unsigned __int64 *)*v37;
                sub_2F07250(v35->m128i_i64, *v37, (__int64)&v37[1][(_QWORD)*v37]);
                v35[2] = _mm_loadu_si128((const __m128i *)v37 + 2);
                a3 = (char *)*((unsigned __int16 *)v37 + 24);
                v35[3].m128i_i16[0] = (__int16)a3;
              }
              v37 += 7;
              v35 = (__m128i *)((char *)v35 + 56);
            }
            while ( v36 != v37 );
          }
          v7[2] = v35;
        }
        v33 += 32;
        v7 += 4;
      }
      while ( v33 != (char *)v42 );
    }
    else
    {
      v9 = v5 + 8;
      v10 = (unsigned __int64 **)(v4 + 1);
      v11 = v6 >> 5;
      if ( v6 > 0 )
      {
        do
        {
          v12 = (__int64)*(v10 - 1);
          v13 = v10;
          v14 = v9;
          v10 += 4;
          v9 += 32;
          *(_QWORD *)(v9 - 40) = v12;
          sub_2F092C0(v14, v13);
          --v11;
        }
        while ( v11 );
        v5 += v40;
      }
      for ( ; v7 != (_QWORD *)v5; v5 += 32LL )
      {
        v15 = *(unsigned __int64 **)(v5 + 16);
        v16 = *(unsigned __int64 **)(v5 + 8);
        if ( v15 != v16 )
        {
          do
          {
            if ( (unsigned __int64 *)*v16 != v16 + 2 )
              j_j___libc_free_0(*v16);
            v16 += 7;
          }
          while ( v15 != v16 );
          v16 = *(unsigned __int64 **)(v5 + 8);
        }
        if ( v16 )
          j_j___libc_free_0((unsigned __int64)v16);
      }
    }
    v17 = *v41 + v40;
LABEL_17:
    v41[1] = v17;
  }
}
