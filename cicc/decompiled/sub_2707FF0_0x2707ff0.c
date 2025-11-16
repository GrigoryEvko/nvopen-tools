// Function: sub_2707FF0
// Address: 0x2707ff0
//
void __fastcall sub_2707FF0(unsigned __int64 *a1)
{
  __int64 v1; // rbx
  __m128i *v2; // rsi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdx
  char v6; // al
  __m128i v7; // xmm0
  char v8; // dl
  bool v9; // dl
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r12
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // rdi
  __int64 v17; // [rsp+10h] [rbp-A0h]
  unsigned __int64 v18; // [rsp+20h] [rbp-90h]
  __m128i v19; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v20; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-68h]
  char v22; // [rsp+50h] [rbp-60h] BYREF
  char v23; // [rsp+78h] [rbp-38h]

  v1 = qword_4FF91C8;
  v17 = qword_4FF91D0;
  if ( qword_4FF91C8 != qword_4FF91D0 )
  {
    while ( 1 )
    {
      v2 = *(__m128i **)v1;
      sub_109B500(&v19, *(_QWORD *)v1, *(_QWORD *)(v1 + 8), v18, 0);
      v5 = v23 & 1;
      v6 = (2 * v5) | v23 & 0xFD;
      v23 = v6;
      if ( (_BYTE)v5 )
LABEL_3:
        sub_F30080(&v19, (__int64)v2);
      v2 = (__m128i *)a1[1];
      if ( v2 == (__m128i *)a1[2] )
        break;
      if ( v2 )
      {
        v7 = _mm_loadu_si128(&v19);
        v2[1].m128i_i64[0] = (__int64)v2[2].m128i_i64;
        v2[1].m128i_i64[1] = 0x100000000LL;
        *v2 = v7;
        if ( v21 )
          sub_2707540((__int64)v2[1].m128i_i64, (__int64)&v20, v5, (__int64)a1, (__int64)&v20, v4);
        v2 = (__m128i *)a1;
        v6 = v23;
        v8 = v23;
        a1[1] += 72LL;
        v9 = (v8 & 2) != 0;
LABEL_9:
        if ( v9 )
          goto LABEL_3;
        goto LABEL_10;
      }
      a1[1] = 72;
LABEL_10:
      if ( (v6 & 1) != 0 )
      {
        if ( v19.m128i_i64[0] )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v19.m128i_i64[0] + 8LL))(v19.m128i_i64[0]);
      }
      else
      {
        v10 = v20;
        v11 = v20 + 40LL * v21;
        if ( v20 != v11 )
        {
          do
          {
            v11 -= 40LL;
            v12 = *(_QWORD *)(v11 + 16);
            if ( v12 != v11 + 40 )
              _libc_free(v12);
            v13 = *(_QWORD *)v11;
            v14 = *(_QWORD *)v11 + 80LL * *(unsigned int *)(v11 + 8);
            if ( *(_QWORD *)v11 != v14 )
            {
              do
              {
                v14 -= 80;
                v15 = *(_QWORD *)(v14 + 8);
                if ( v15 != v14 + 24 )
                  _libc_free(v15);
              }
              while ( v13 != v14 );
              v13 = *(_QWORD *)v11;
            }
            if ( v13 != v11 + 16 )
              _libc_free(v13);
          }
          while ( v10 != v11 );
          v11 = v20;
        }
        if ( (char *)v11 != &v22 )
          _libc_free(v11);
      }
      v1 += 32;
      if ( v17 == v1 )
        return;
    }
    sub_2707CD0(a1, v2, &v19, (__int64)a1, v3, v4);
    v6 = v23;
    v9 = (v23 & 2) != 0;
    goto LABEL_9;
  }
}
