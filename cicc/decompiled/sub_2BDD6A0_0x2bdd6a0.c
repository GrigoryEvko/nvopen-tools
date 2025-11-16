// Function: sub_2BDD6A0
// Address: 0x2bdd6a0
//
__int64 __fastcall sub_2BDD6A0(unsigned __int64 **a1, const __m128i **a2, int a3)
{
  unsigned __int64 *v5; // r13
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r12
  __int64 v10; // rdi
  const __m128i *v11; // r14
  __m128i *v12; // rax
  unsigned __int16 *v13; // rdx
  __m128i *v14; // r13
  signed __int64 v15; // r12
  char *v16; // rcx
  __int64 v17; // r12
  size_t v18; // rax
  unsigned __int64 v19; // r15
  __int64 *v20; // r12
  __int64 v21; // r15
  __int64 i; // rax
  unsigned __int64 v23; // r12
  _WORD *v24; // rax
  unsigned __int16 *v25; // rcx
  _WORD *v26; // rcx
  unsigned __int64 v27; // r12
  _DWORD *v28; // rax
  char *v29; // rcx
  char *v30; // rdx
  _DWORD *v31; // rcx
  __m128i v32; // xmm0
  __m128i v33; // xmm1
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-38h]

  switch ( a3 )
  {
    case 1:
      *a1 = (unsigned __int64 *)*a2;
      return 0;
    case 2:
      v10 = 160;
      v11 = *a2;
      v12 = (__m128i *)sub_22077B0(0xA0u);
      v14 = v12;
      if ( !v12 )
      {
LABEL_51:
        *a1 = (unsigned __int64 *)v14;
        return 0;
      }
      v15 = v11->m128i_i64[1] - v11->m128i_i64[0];
      v12->m128i_i64[0] = 0;
      v12->m128i_i64[1] = 0;
      v12[1].m128i_i64[0] = 0;
      if ( v15 )
      {
        if ( v15 < 0 )
          goto LABEL_58;
        v10 = v15;
        v16 = (char *)sub_22077B0(v15);
      }
      else
      {
        v16 = 0;
      }
      v14->m128i_i64[0] = (__int64)v16;
      v14[1].m128i_i64[0] = (__int64)&v16[v15];
      v17 = 0;
      v14->m128i_i64[1] = (__int64)v16;
      a2 = (const __m128i **)v11->m128i_i64[0];
      v18 = v11->m128i_i64[1] - v11->m128i_i64[0];
      if ( v18 )
      {
        v10 = (__int64)v16;
        v17 = v11->m128i_i64[1] - v11->m128i_i64[0];
        v16 = (char *)memmove(v16, a2, v18);
      }
      v14->m128i_i64[1] = (__int64)&v16[v17];
      v19 = v11[2].m128i_i64[0] - v11[1].m128i_i64[1];
      v14[1].m128i_i64[1] = 0;
      v14[2].m128i_i64[0] = 0;
      v14[2].m128i_i64[1] = 0;
      if ( v19 )
      {
        if ( v19 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_58;
        v10 = v19;
        v20 = (__int64 *)sub_22077B0(v19);
      }
      else
      {
        v19 = 0;
        v20 = 0;
      }
      v14[1].m128i_i64[1] = (__int64)v20;
      v14[2].m128i_i64[0] = (__int64)v20;
      v14[2].m128i_i64[1] = (__int64)v20 + v19;
      v21 = v11[2].m128i_i64[0];
      for ( i = v11[1].m128i_i64[1]; v21 != i; v20 += 4 )
      {
        if ( v20 )
        {
          v10 = (__int64)v20;
          v36 = i;
          *v20 = (__int64)(v20 + 2);
          a2 = *(const __m128i ***)i;
          sub_2BDC2F0(v20, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
          i = v36;
        }
        i += 32;
      }
      v14[2].m128i_i64[0] = (__int64)v20;
      v23 = v11[3].m128i_i64[1] - v11[3].m128i_i64[0];
      v14[3].m128i_i64[0] = 0;
      v14[3].m128i_i64[1] = 0;
      v14[4].m128i_i64[0] = 0;
      if ( v23 )
      {
        if ( v23 > 0x7FFFFFFFFFFFFFFELL )
          goto LABEL_58;
        v10 = v23;
        v24 = (_WORD *)sub_22077B0(v23);
      }
      else
      {
        v23 = 0;
        v24 = 0;
      }
      v14[3].m128i_i64[0] = (__int64)v24;
      v14[3].m128i_i64[1] = (__int64)v24;
      v14[4].m128i_i64[0] = (__int64)v24 + v23;
      v25 = (unsigned __int16 *)v11[3].m128i_i64[1];
      v13 = (unsigned __int16 *)v11[3].m128i_i64[0];
      if ( v25 == v13 )
      {
        v26 = v24;
      }
      else
      {
        v26 = (_WORD *)((char *)v24 + (char *)v25 - (char *)v13);
        do
        {
          if ( v24 )
          {
            a2 = (const __m128i **)*v13;
            *v24 = (_WORD)a2;
          }
          ++v24;
          ++v13;
        }
        while ( v24 != v26 );
      }
      v14[3].m128i_i64[1] = (__int64)v26;
      v27 = v11[5].m128i_i64[0] - v11[4].m128i_i64[1];
      v14[4].m128i_i64[1] = 0;
      v14[5].m128i_i64[0] = 0;
      v14[5].m128i_i64[1] = 0;
      if ( !v27 )
      {
        v27 = 0;
        v28 = 0;
        goto LABEL_45;
      }
      if ( v27 <= 0x7FFFFFFFFFFFFFFCLL )
      {
        v28 = (_DWORD *)sub_22077B0(v27);
LABEL_45:
        v14[4].m128i_i64[1] = (__int64)v28;
        v14[5].m128i_i64[0] = (__int64)v28;
        v14[5].m128i_i64[1] = (__int64)v28 + v27;
        v29 = (char *)v11[5].m128i_i64[0];
        v30 = (char *)v11[4].m128i_i64[1];
        if ( v29 == v30 )
        {
          v31 = v28;
        }
        else
        {
          v31 = (_DWORD *)((char *)v28 + v29 - v30);
          do
          {
            if ( v28 )
              *v28 = *(_DWORD *)v30;
            ++v28;
            v30 += 4;
          }
          while ( v28 != v31 );
        }
        v14[5].m128i_i64[0] = (__int64)v31;
        v32 = _mm_loadu_si128(v11 + 8);
        v33 = _mm_loadu_si128(v11 + 9);
        v14[6].m128i_i32[0] = v11[6].m128i_i32[0];
        v34 = v11[6].m128i_i64[1];
        v14[8] = v32;
        v14[6].m128i_i64[1] = v34;
        v35 = v11[7].m128i_i64[0];
        v14[9] = v33;
        v14[7].m128i_i64[0] = v35;
        v14[7].m128i_i8[8] = v11[7].m128i_i8[8];
        goto LABEL_51;
      }
LABEL_58:
      sub_4261EA(v10, a2, v13);
    case 3:
      v5 = *a1;
      if ( *a1 )
      {
        v6 = v5[9];
        if ( v6 )
          j_j___libc_free_0(v6);
        v7 = v5[6];
        if ( v7 )
          j_j___libc_free_0(v7);
        v8 = (unsigned __int64 *)v5[4];
        v9 = (unsigned __int64 *)v5[3];
        if ( v8 != v9 )
        {
          do
          {
            if ( (unsigned __int64 *)*v9 != v9 + 2 )
              j_j___libc_free_0(*v9);
            v9 += 4;
          }
          while ( v8 != v9 );
          v9 = (unsigned __int64 *)v5[3];
        }
        if ( v9 )
          j_j___libc_free_0((unsigned __int64)v9);
        if ( *v5 )
          j_j___libc_free_0(*v5);
        j_j___libc_free_0((unsigned __int64)v5);
      }
      break;
  }
  return 0;
}
