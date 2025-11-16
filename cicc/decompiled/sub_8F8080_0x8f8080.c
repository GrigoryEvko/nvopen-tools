// Function: sub_8F8080
// Address: 0x8f8080
//
__int64 __fastcall sub_8F8080(char *s, int a2, unsigned __int8 *a3)
{
  int v3; // r12d
  char *v4; // r13
  __int64 result; // rax
  int v6; // eax
  __int64 v7; // r9
  __int64 v8; // r10
  unsigned __int64 v9; // rax
  int v10; // ecx
  __m128i v11; // xmm5
  signed __int64 v12; // rcx
  __int64 *v13; // rcx
  __int64 v14; // rcx
  const __m128i *v15; // rax
  __m128i si128; // xmm7
  __m128i v17; // xmm1
  int v18; // eax
  __m128i v19; // xmm3
  __int64 v20; // rax
  int v21; // edx
  int v22; // [rsp-108h] [rbp-108h]
  char v23; // [rsp-108h] [rbp-108h]
  int v24; // [rsp-104h] [rbp-104h]
  __int64 i; // [rsp-100h] [rbp-100h]
  __m128i v26; // [rsp-F8h] [rbp-F8h] BYREF
  __m128i v27; // [rsp-E8h] [rbp-E8h] BYREF
  __m128i v28; // [rsp-D8h] [rbp-D8h] BYREF
  __m128i v29; // [rsp-C8h] [rbp-C8h] BYREF
  __m128i v30; // [rsp-B8h] [rbp-B8h] BYREF
  __m128i v31; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v32; // [rsp-98h] [rbp-98h] BYREF
  __m128i v33; // [rsp-88h] [rbp-88h]
  _DWORD v34[30]; // [rsp-78h] [rbp-78h] BYREF

  if ( !unk_4D04520 )
    JUMPOUT(0x8F70F0);
  if ( !s )
    return 4294967294LL;
  v3 = a2;
  if ( a2 <= 0 )
    return 4294967294LL;
  if ( !a3 )
    return 4294967293LL;
  v4 = s;
  sub_8EF310((__int64)&v26, a3);
  if ( v26.m128i_i32[1] && v26.m128i_i32[0] != 3 && v26.m128i_i32[0] )
  {
    *s = 45;
    v3 = a2 - 1;
    v4 = s + 1;
  }
  result = sub_8F1A40(v4, v3, v26.m128i_i32);
  if ( (_DWORD)result == -1 )
  {
    if ( v26.m128i_i32[2] <= 0 )
    {
      if ( v26.m128i_i32[2] < -16381 )
      {
        v10 = -16381 - v26.m128i_i32[2];
        goto LABEL_15;
      }
    }
    else
    {
      v6 = sub_8EE460((__int64)&v26.m128i_i64[1] + 4, v27.m128i_i32[3]);
      if ( (int)v8 >= (int)v7 - v6 )
      {
        v9 = 30103 * v8;
        if ( 30103 * v8 <= 4899999 )
        {
          v11 = _mm_loadu_si128(&v27);
          v28 = _mm_loadu_si128(&v26);
          v29 = v11;
          v28.m128i_i32[1] = 0;
          v12 = v9 / 0x186A0;
          v24 = v9 / 0x186A0;
          if ( v9 / 0x186A0 )
          {
            v13 = &qword_4F612E0[4 * (int)v12];
            do
            {
              if ( !sub_8F0EB0(&v28, v13) )
              {
                v12 = v24;
                goto LABEL_24;
              }
              v13 = (__int64 *)(v14 - 32);
              --v24;
            }
            while ( v24 );
            v12 = 0;
          }
          else
          {
            v24 = 0;
          }
LABEL_24:
          if ( 30103 * v7 / 100000 > v12 )
          {
            v15 = (const __m128i *)&qword_4F612E0[4 * v24];
            si128 = _mm_load_si128(v15 + 1);
            v30 = _mm_load_si128(v15);
            v31 = si128;
            if ( v28.m128i_i32[0] == 6 )
            {
              v22 = 0;
            }
            else
            {
              for ( i = 0; ; ++i )
              {
                v17 = _mm_loadu_si128(&v29);
                v32 = _mm_loadu_si128(&v28);
                v33 = v17;
                sub_8F0F10(&v32, (__int64)&v30);
                v18 = sub_8EEF20((__int64)&v32);
                v19 = _mm_loadu_si128(&v31);
                v23 = v18;
                v32 = _mm_loadu_si128(&v30);
                v33 = v19;
                if ( v18 )
                {
                  if ( v32.m128i_i32[0] != 6 )
                    sub_8F06E0(&v32, v18);
                }
                else
                {
                  v32.m128i_i32[0] = 6;
                }
                if ( sub_8F0EB0(&v28, &v32) )
                {
                  --v23;
                  sub_8EF5D0(&v32, &v30);
                }
                sub_8EF5D0(&v28, &v32);
                *((_BYTE *)&v34[2] + i) = v23 + 48;
                v22 = i + 1;
                if ( v28.m128i_i32[0] == 6 )
                  break;
                sub_8F06E0(&v28, 10);
                if ( v28.m128i_i32[0] == 6 )
                  break;
              }
            }
            v20 = v22;
            do
            {
              v21 = v20;
              if ( (int)v20 <= 1 )
                break;
              --v20;
            }
            while ( *((_BYTE *)&v34[2] + v20) == 48 );
            v34[1] = v24 + 1;
            v34[12] = v21;
            v34[0] = v26.m128i_i32[1];
            *((_BYTE *)&v34[2] + v21) = 0;
            return sub_8EFB80(v4, v3, (__int64)v34);
          }
        }
      }
    }
    v10 = 0;
LABEL_15:
    sub_8F2910((__int64)v34, (__int64)&v26, 0x7FFFFFFF, v10);
    return sub_8EFB80(v4, v3, (__int64)v34);
  }
  return result;
}
