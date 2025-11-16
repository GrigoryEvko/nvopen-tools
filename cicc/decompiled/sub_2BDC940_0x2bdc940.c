// Function: sub_2BDC940
// Address: 0x2bdc940
//
__int64 __fastcall sub_2BDC940(unsigned __int64 **a1, const __m128i **a2, int a3)
{
  unsigned __int64 *v4; // r13
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 *v9; // rbx
  unsigned __int64 *v10; // r12
  __int64 v11; // rdi
  const __m128i *v12; // r13
  __m128i *v13; // rax
  __int64 v14; // rdx
  __m128i *v15; // r14
  signed __int64 v16; // rbx
  char *v17; // rdi
  __int64 v18; // rbx
  size_t v19; // rax
  unsigned __int64 v20; // r12
  __int64 *v21; // rbx
  __int64 v22; // r12
  __int64 i; // r15
  unsigned __int64 v24; // r12
  __int64 *v25; // rbx
  __int64 v26; // r15
  __int64 j; // r12
  unsigned __int64 v28; // rbx
  _DWORD *v29; // rax
  char *v30; // rsi
  char *v31; // rdx
  _DWORD *v32; // rsi
  __m128i v33; // xmm0
  __m128i v34; // xmm1
  __int64 v35; // rax
  __int64 v36; // rax

  switch ( a3 )
  {
    case 1:
      *a1 = (unsigned __int64 *)*a2;
      return 0;
    case 2:
      v11 = 160;
      v12 = *a2;
      v13 = (__m128i *)sub_22077B0(0xA0u);
      v15 = v13;
      if ( !v13 )
      {
LABEL_57:
        *a1 = (unsigned __int64 *)v15;
        return 0;
      }
      v16 = v12->m128i_i64[1] - v12->m128i_i64[0];
      v13->m128i_i64[0] = 0;
      v13->m128i_i64[1] = 0;
      v13[1].m128i_i64[0] = 0;
      if ( v16 )
      {
        if ( v16 < 0 )
          goto LABEL_63;
        v17 = (char *)sub_22077B0(v16);
      }
      else
      {
        v17 = 0;
      }
      v15->m128i_i64[0] = (__int64)v17;
      v15[1].m128i_i64[0] = (__int64)&v17[v16];
      v18 = 0;
      v15->m128i_i64[1] = (__int64)v17;
      a2 = (const __m128i **)v12->m128i_i64[0];
      v19 = v12->m128i_i64[1] - v12->m128i_i64[0];
      if ( v19 )
      {
        v18 = v12->m128i_i64[1] - v12->m128i_i64[0];
        v17 = (char *)memmove(v17, a2, v19);
      }
      v11 = (__int64)&v17[v18];
      v15->m128i_i64[1] = v11;
      v20 = v12[2].m128i_i64[0] - v12[1].m128i_i64[1];
      v15[1].m128i_i64[1] = 0;
      v15[2].m128i_i64[0] = 0;
      v15[2].m128i_i64[1] = 0;
      if ( v20 )
      {
        if ( v20 > 0x7FFFFFFFFFFFFFE0LL )
          goto LABEL_63;
        v11 = v20;
        v21 = (__int64 *)sub_22077B0(v20);
      }
      else
      {
        v20 = 0;
        v21 = 0;
      }
      v15[1].m128i_i64[1] = (__int64)v21;
      v15[2].m128i_i64[0] = (__int64)v21;
      v15[2].m128i_i64[1] = (__int64)v21 + v20;
      v22 = v12[2].m128i_i64[0];
      for ( i = v12[1].m128i_i64[1]; v22 != i; v21 += 4 )
      {
        if ( v21 )
        {
          v11 = (__int64)v21;
          *v21 = (__int64)(v21 + 2);
          a2 = *(const __m128i ***)i;
          sub_2BDC2F0(v21, *(_BYTE **)i, *(_QWORD *)i + *(_QWORD *)(i + 8));
        }
        i += 32;
      }
      v15[2].m128i_i64[0] = (__int64)v21;
      v24 = v12[3].m128i_i64[1] - v12[3].m128i_i64[0];
      v15[3].m128i_i64[0] = 0;
      v15[3].m128i_i64[1] = 0;
      v15[4].m128i_i64[0] = 0;
      if ( v24 )
      {
        if ( v24 > 0x7FFFFFFFFFFFFFC0LL )
          goto LABEL_63;
        v11 = v24;
        v25 = (__int64 *)sub_22077B0(v24);
      }
      else
      {
        v24 = 0;
        v25 = 0;
      }
      v15[3].m128i_i64[0] = (__int64)v25;
      v15[3].m128i_i64[1] = (__int64)v25;
      v15[4].m128i_i64[0] = (__int64)v25 + v24;
      v26 = v12[3].m128i_i64[1];
      for ( j = v12[3].m128i_i64[0]; v26 != j; v25 += 8 )
      {
        if ( v25 )
        {
          *v25 = (__int64)(v25 + 2);
          sub_2BDC2F0(v25, *(_BYTE **)j, *(_QWORD *)j + *(_QWORD *)(j + 8));
          v11 = (__int64)(v25 + 4);
          v25[4] = (__int64)(v25 + 6);
          a2 = *(const __m128i ***)(j + 32);
          sub_2BDC2F0(v25 + 4, a2, (__int64)a2 + *(_QWORD *)(j + 40));
        }
        j += 64;
      }
      v15[3].m128i_i64[1] = (__int64)v25;
      v28 = v12[5].m128i_i64[0] - v12[4].m128i_i64[1];
      v15[4].m128i_i64[1] = 0;
      v15[5].m128i_i64[0] = 0;
      v15[5].m128i_i64[1] = 0;
      if ( !v28 )
      {
        v28 = 0;
        v29 = 0;
        goto LABEL_51;
      }
      if ( v28 <= 0x7FFFFFFFFFFFFFFCLL )
      {
        v29 = (_DWORD *)sub_22077B0(v28);
LABEL_51:
        v15[4].m128i_i64[1] = (__int64)v29;
        v15[5].m128i_i64[0] = (__int64)v29;
        v15[5].m128i_i64[1] = (__int64)v29 + v28;
        v30 = (char *)v12[5].m128i_i64[0];
        v31 = (char *)v12[4].m128i_i64[1];
        if ( v30 == v31 )
        {
          v32 = v29;
        }
        else
        {
          v32 = (_DWORD *)((char *)v29 + v30 - v31);
          do
          {
            if ( v29 )
              *v29 = *(_DWORD *)v31;
            ++v29;
            v31 += 4;
          }
          while ( v29 != v32 );
        }
        v15[5].m128i_i64[0] = (__int64)v32;
        v33 = _mm_loadu_si128(v12 + 8);
        v34 = _mm_loadu_si128(v12 + 9);
        v15[6].m128i_i32[0] = v12[6].m128i_i32[0];
        v35 = v12[6].m128i_i64[1];
        v15[8] = v33;
        v15[6].m128i_i64[1] = v35;
        v36 = v12[7].m128i_i64[0];
        v15[9] = v34;
        v15[7].m128i_i64[0] = v36;
        v15[7].m128i_i8[8] = v12[7].m128i_i8[8];
        goto LABEL_57;
      }
LABEL_63:
      sub_4261EA(v11, a2, v14);
    case 3:
      v4 = *a1;
      if ( *a1 )
      {
        v5 = v4[9];
        if ( v5 )
          j_j___libc_free_0(v5);
        v6 = (unsigned __int64 *)v4[7];
        v7 = (unsigned __int64 *)v4[6];
        if ( v6 != v7 )
        {
          do
          {
            v8 = v7[4];
            if ( (unsigned __int64 *)v8 != v7 + 6 )
              j_j___libc_free_0(v8);
            if ( (unsigned __int64 *)*v7 != v7 + 2 )
              j_j___libc_free_0(*v7);
            v7 += 8;
          }
          while ( v6 != v7 );
          v7 = (unsigned __int64 *)v4[6];
        }
        if ( v7 )
          j_j___libc_free_0((unsigned __int64)v7);
        v9 = (unsigned __int64 *)v4[4];
        v10 = (unsigned __int64 *)v4[3];
        if ( v9 != v10 )
        {
          do
          {
            if ( (unsigned __int64 *)*v10 != v10 + 2 )
              j_j___libc_free_0(*v10);
            v10 += 4;
          }
          while ( v9 != v10 );
          v10 = (unsigned __int64 *)v4[3];
        }
        if ( v10 )
          j_j___libc_free_0((unsigned __int64)v10);
        if ( *v4 )
          j_j___libc_free_0(*v4);
        j_j___libc_free_0((unsigned __int64)v4);
      }
      break;
  }
  return 0;
}
