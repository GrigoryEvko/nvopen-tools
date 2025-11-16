// Function: sub_2BB8B30
// Address: 0x2bb8b30
//
void __fastcall sub_2BB8B30(__int64 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 i; // rbx
  __m128i v9; // xmm0
  __int64 m128i_i64; // r13
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdi
  unsigned __int64 *v18; // rdx
  unsigned __int64 *v19; // r13
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdi
  __int32 v25; // eax
  __int64 v26; // rcx
  __int64 v27; // r13
  __int64 v28; // r12
  __int64 v29; // rbx
  unsigned __int64 v30; // r14
  unsigned __int64 *v31; // rbx
  __int64 v32; // [rsp+8h] [rbp-58h]
  unsigned int v33; // [rsp+8h] [rbp-58h]
  __int32 v34; // [rsp+8h] [rbp-58h]
  unsigned int v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  __int64 v37; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v38; // [rsp+18h] [rbp-48h]
  __int64 v39; // [rsp+18h] [rbp-48h]
  unsigned int v40; // [rsp+18h] [rbp-48h]
  unsigned __int64 v41[7]; // [rsp+28h] [rbp-38h] BYREF

  v5 = *a1 + 96LL * *((unsigned int *)a1 + 2);
  if ( *a1 != v5 )
  {
    for ( i = *a1 + 32; ; i += 96 )
    {
      if ( a2 )
      {
        v9 = _mm_loadu_si128((const __m128i *)(i - 32));
        m128i_i64 = (__int64)a2[2].m128i_i64;
        a2[1].m128i_i32[2] = 0;
        a2[1].m128i_i64[0] = (__int64)a2[2].m128i_i64;
        a2[1].m128i_i32[3] = 1;
        *a2 = v9;
        v11 = *(unsigned int *)(i - 8);
        if ( (_DWORD)v11 )
        {
          if ( &a2[1] != (__m128i *)(i - 16) )
          {
            v12 = *(_QWORD *)(i - 16);
            if ( v12 == i )
            {
              v13 = i;
              v14 = 1;
              if ( (_DWORD)v11 != 1 )
              {
                v33 = *(_DWORD *)(i - 8);
                v39 = sub_C8D7D0((__int64)a2[1].m128i_i64, (__int64)a2[2].m128i_i64, (unsigned int)v11, 0x40u, v41, v11);
                sub_2BB7D80((__int64)a2[1].m128i_i64, v39, v20, v21, v22, v23);
                v24 = a2[1].m128i_u64[0];
                v25 = v41[0];
                v26 = v39;
                v11 = v33;
                if ( m128i_i64 != v24 )
                {
                  v34 = v41[0];
                  v36 = v39;
                  v40 = v11;
                  _libc_free(v24);
                  v25 = v34;
                  v26 = v36;
                  v11 = v40;
                }
                a2[1].m128i_i64[0] = v26;
                m128i_i64 = v26;
                a2[1].m128i_i32[3] = v25;
                v13 = *(_QWORD *)(i - 16);
                v14 = *(unsigned int *)(i - 8);
              }
              v15 = v13 + (v14 << 6);
              if ( v13 != v15 )
              {
                do
                {
                  while ( 1 )
                  {
                    if ( m128i_i64 )
                    {
                      *(_DWORD *)(m128i_i64 + 8) = 0;
                      *(_QWORD *)m128i_i64 = m128i_i64 + 16;
                      *(_DWORD *)(m128i_i64 + 12) = 3;
                      v16 = *(unsigned int *)(v13 + 8);
                      if ( (_DWORD)v16 )
                        break;
                    }
                    v13 += 64;
                    m128i_i64 += 64;
                    if ( v15 == v13 )
                      goto LABEL_16;
                  }
                  v17 = m128i_i64;
                  v32 = v15;
                  m128i_i64 += 64;
                  v35 = v11;
                  v37 = v13;
                  sub_2BB7BD0(v17, (unsigned __int64 *)v13, v15, v16, a5, v11);
                  v15 = v32;
                  v11 = v35;
                  v13 = v37 + 64;
                }
                while ( v32 != v37 + 64 );
              }
LABEL_16:
              a2[1].m128i_i32[2] = v11;
              v18 = *(unsigned __int64 **)(i - 16);
              v19 = &v18[8 * (unsigned __int64)*(unsigned int *)(i - 8)];
              while ( v18 != v19 )
              {
                v19 -= 8;
                if ( (unsigned __int64 *)*v19 != v19 + 2 )
                {
                  v38 = v18;
                  _libc_free(*v19);
                  v18 = v38;
                }
              }
              *(_DWORD *)(i - 8) = 0;
            }
            else
            {
              a2[1].m128i_i64[0] = v12;
              a2[1].m128i_i32[2] = *(_DWORD *)(i - 8);
              a2[1].m128i_i32[3] = *(_DWORD *)(i - 4);
              *(_QWORD *)(i - 16) = i;
              *(_DWORD *)(i - 4) = 0;
              *(_DWORD *)(i - 8) = 0;
            }
          }
        }
      }
      a2 += 6;
      if ( v5 == i + 64 )
        break;
    }
    v27 = *a1;
    v28 = *a1 + 96LL * *((unsigned int *)a1 + 2);
    while ( v28 != v27 )
    {
      v29 = *(unsigned int *)(v28 - 72);
      v30 = *(_QWORD *)(v28 - 80);
      v28 -= 96;
      v31 = (unsigned __int64 *)(v30 + (v29 << 6));
      if ( (unsigned __int64 *)v30 != v31 )
      {
        do
        {
          v31 -= 8;
          if ( (unsigned __int64 *)*v31 != v31 + 2 )
            _libc_free(*v31);
        }
        while ( (unsigned __int64 *)v30 != v31 );
        v30 = *(_QWORD *)(v28 + 16);
      }
      if ( v30 != v28 + 32 )
        _libc_free(v30);
    }
  }
}
