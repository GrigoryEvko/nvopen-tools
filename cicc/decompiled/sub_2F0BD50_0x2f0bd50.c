// Function: sub_2F0BD50
// Address: 0x2f0bd50
//
void __fastcall sub_2F0BD50(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  const __m128i *v4; // r14
  const __m128i *v5; // rbx
  __m128i *i; // r15
  const __m128i *v7; // r14
  const __m128i *v8; // rbx
  __int64 v9; // rax
  __m128i *j; // r13
  bool v11; // zf
  unsigned __int64 *k; // rbx
  unsigned __int64 *m; // r14
  _QWORD *v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  unsigned __int64 *v20; // rbx
  unsigned __int64 v21; // r12
  __int64 v22; // rax
  unsigned __int64 *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  unsigned __int64 v27; // r12
  unsigned __int64 *v28; // rbx
  __int64 v29; // r13
  unsigned __int64 *v30; // r14
  __int64 v31; // [rsp+8h] [rbp-78h]
  unsigned int v32; // [rsp+14h] [rbp-6Ch]
  __int64 v33; // [rsp+18h] [rbp-68h]
  __int64 v34; // [rsp+18h] [rbp-68h]
  unsigned int v35; // [rsp+20h] [rbp-60h]
  __int64 v36; // [rsp+20h] [rbp-60h]
  unsigned __int64 v37; // [rsp+28h] [rbp-58h]
  __int64 v38; // [rsp+28h] [rbp-58h]
  unsigned int v39; // [rsp+30h] [rbp-50h]
  unsigned int v40; // [rsp+30h] [rbp-50h]
  unsigned int v41; // [rsp+34h] [rbp-4Ch]
  bool v42; // [rsp+34h] [rbp-4Ch]
  int v43; // [rsp+34h] [rbp-4Ch]
  unsigned __int64 v44; // [rsp+38h] [rbp-48h]
  unsigned __int64 v45; // [rsp+40h] [rbp-40h]
  unsigned __int64 v46; // [rsp+40h] [rbp-40h]
  unsigned __int64 v47; // [rsp+48h] [rbp-38h]
  __int64 v48; // [rsp+48h] [rbp-38h]

  v44 = a1;
  v31 = a2;
  if ( a1 != a2 )
  {
    v3 = a1 + 32;
    if ( a1 + 32 != a2 )
    {
      while ( 1 )
      {
        v4 = *(const __m128i **)(v44 + 16);
        v5 = *(const __m128i **)(v44 + 8);
        v41 = *(_DWORD *)v44;
        v39 = *(_DWORD *)(v44 + 4);
        v37 = (char *)v4 - (char *)v5;
        if ( v4 == v5 )
        {
          v47 = 0;
        }
        else
        {
          if ( (unsigned __int64)((char *)v4 - (char *)v5) > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_54;
          a1 = (char *)v4 - (char *)v5;
          v47 = sub_22077B0(v37);
          v4 = *(const __m128i **)(v44 + 16);
          v5 = *(const __m128i **)(v44 + 8);
        }
        for ( i = (__m128i *)v47; v4 != v5; i = (__m128i *)((char *)i + 56) )
        {
          if ( i )
          {
            a1 = (unsigned __int64)i;
            i->m128i_i64[0] = (__int64)i[1].m128i_i64;
            a2 = v5->m128i_i64[0];
            sub_2F07250(i->m128i_i64, v5->m128i_i64[0], v5->m128i_i64[0] + v5->m128i_i64[1]);
            i[2] = _mm_loadu_si128(v5 + 2);
            i[3].m128i_i16[0] = v5[3].m128i_i16[0];
          }
          v5 = (const __m128i *)((char *)v5 + 56);
        }
        v7 = *(const __m128i **)(v3 + 16);
        v8 = *(const __m128i **)(v3 + 8);
        v35 = *(_DWORD *)v3;
        v32 = *(_DWORD *)(v3 + 4);
        v33 = (char *)v7 - (char *)v8;
        if ( v7 == v8 )
        {
          v45 = 0;
        }
        else
        {
          if ( (unsigned __int64)((char *)v7 - (char *)v8) > 0x7FFFFFFFFFFFFFF8LL )
LABEL_54:
            sub_4261EA(a1, a2, a3);
          a1 = (char *)v7 - (char *)v8;
          v9 = sub_22077B0((char *)v7 - (char *)v8);
          v7 = *(const __m128i **)(v3 + 16);
          v8 = *(const __m128i **)(v3 + 8);
          v45 = v9;
        }
        for ( j = (__m128i *)v45; v7 != v8; j = (__m128i *)((char *)j + 56) )
        {
          if ( j )
          {
            a1 = (unsigned __int64)j;
            j->m128i_i64[0] = (__int64)j[1].m128i_i64;
            a2 = v8->m128i_i64[0];
            sub_2F07250(j->m128i_i64, v8->m128i_i64[0], v8->m128i_i64[0] + v8->m128i_i64[1]);
            j[2] = _mm_loadu_si128(v8 + 2);
            j[3].m128i_i16[0] = v8[3].m128i_i16[0];
          }
          v8 = (const __m128i *)((char *)v8 + 56);
        }
        v11 = v41 == v35;
        v42 = v41 > v35;
        if ( v11 )
          v42 = v39 > v32;
        for ( k = (unsigned __int64 *)v45; k != (unsigned __int64 *)j; k += 7 )
        {
          a1 = *k;
          if ( (unsigned __int64 *)*k != k + 2 )
          {
            a2 = k[2] + 1;
            j_j___libc_free_0(a1);
          }
        }
        if ( v45 )
        {
          a2 = v33;
          a1 = v45;
          j_j___libc_free_0(v45);
        }
        for ( m = (unsigned __int64 *)v47; m != (unsigned __int64 *)i; m += 7 )
        {
          a1 = *m;
          if ( (unsigned __int64 *)*m != m + 2 )
          {
            a2 = m[2] + 1;
            j_j___libc_free_0(a1);
          }
        }
        if ( v47 )
        {
          a2 = v37;
          a1 = v47;
          j_j___libc_free_0(v47);
        }
        v14 = (_QWORD *)(v3 + 32);
        v46 = v3 + 32;
        if ( v42 )
        {
          v40 = *(_DWORD *)v3;
          v43 = *(_DWORD *)(v3 + 4);
          v15 = *(_QWORD *)(v3 + 8);
          *(_QWORD *)(v3 + 8) = 0;
          v38 = v15;
          v16 = *(_QWORD *)(v3 + 16);
          *(_QWORD *)(v3 + 16) = 0;
          v36 = v16;
          v17 = *(_QWORD *)(v3 + 24);
          *(_QWORD *)(v3 + 24) = 0;
          v18 = v3 - v44;
          v34 = v17;
          v19 = v18 >> 5;
          if ( v18 > 0 )
          {
            v48 = 0;
            v20 = 0;
            v21 = 0;
            while ( 1 )
            {
              v22 = *(v14 - 8);
              v14 -= 4;
              v23 = (unsigned __int64 *)v21;
              *v14 = v22;
              v24 = *(v14 - 3);
              *(v14 - 3) = 0;
              v14[1] = v24;
              v25 = *(v14 - 2);
              *(v14 - 2) = 0;
              v14[2] = v25;
              v26 = *(v14 - 1);
              *(v14 - 1) = 0;
              for ( v14[3] = v26; v23 != v20; v23 += 7 )
              {
                a1 = *v23;
                if ( (unsigned __int64 *)*v23 != v23 + 2 )
                {
                  a2 = v23[2] + 1;
                  j_j___libc_free_0(a1);
                }
              }
              if ( v21 )
              {
                a1 = v21;
                a2 = v48 - v21;
                j_j___libc_free_0(v21);
              }
              if ( !--v19 )
                break;
              v21 = *(v14 - 3);
              v20 = (unsigned __int64 *)*(v14 - 2);
              v48 = *(v14 - 1);
            }
          }
          v27 = *(_QWORD *)(v44 + 8);
          v28 = *(unsigned __int64 **)(v44 + 16);
          *(_DWORD *)v44 = v40;
          v29 = *(_QWORD *)(v44 + 24);
          v30 = (unsigned __int64 *)v27;
          *(_DWORD *)(v44 + 4) = v43;
          *(_QWORD *)(v44 + 8) = v38;
          *(_QWORD *)(v44 + 16) = v36;
          for ( *(_QWORD *)(v44 + 24) = v34; v28 != v30; v30 += 7 )
          {
            a1 = *v30;
            if ( (unsigned __int64 *)*v30 != v30 + 2 )
            {
              a2 = v30[2] + 1;
              j_j___libc_free_0(a1);
            }
          }
          if ( v27 )
          {
            a1 = v27;
            a2 = v29 - v27;
            j_j___libc_free_0(v27);
          }
          if ( v31 == v46 )
            return;
        }
        else
        {
          a1 = v3;
          sub_2F0B9C0(v3, a2, a3);
          if ( v31 == v46 )
            return;
        }
        v3 = v46;
      }
    }
  }
}
