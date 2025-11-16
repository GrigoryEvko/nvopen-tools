// Function: sub_C302C0
// Address: 0xc302c0
//
__int64 __fastcall sub_C302C0(__int64 *a1, __int64 a2, __m128i *a3, const __m128i *a4, unsigned __int8 a5)
{
  __int64 v7; // rax
  unsigned __int8 v8; // r13
  __int64 result; // rax
  __m128i v10; // xmm2
  __int32 v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  char v18; // [rsp+1Ah] [rbp-76h] BYREF
  char v19; // [rsp+1Bh] [rbp-75h] BYREF
  __int32 v20; // [rsp+1Ch] [rbp-74h] BYREF
  __int32 v21; // [rsp+20h] [rbp-70h] BYREF
  int v22; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  __m128i v24; // [rsp+30h] [rbp-60h] BYREF
  __int64 v25; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int64 v26; // [rsp+48h] [rbp-48h]

  v7 = *a1;
  v18 = 1;
  v8 = (*(__int64 (__fastcall **)(__int64 *))(v7 + 16))(a1);
  if ( v8 )
    v8 = a3[1].m128i_i8[8] ^ 1;
  result = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  if ( (_BYTE)result )
  {
    if ( !a3[1].m128i_i8[8] )
      goto LABEL_19;
  }
  else if ( !a3[1].m128i_i8[8] )
  {
    a3->m128i_i64[0] = 0;
    a3->m128i_i64[1] = 0;
    a3[1].m128i_i64[0] = 0;
    a3[1].m128i_i8[8] = 1;
  }
  result = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD, _QWORD, char *, __int64 *))(*a1 + 120))(
             a1,
             a2,
             a5,
             v8,
             &v18,
             &v23);
  if ( (_BYTE)result )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
      goto LABEL_8;
    v13 = sub_CB1000(a1);
    if ( *(_DWORD *)(v13 + 32) != 1 )
      goto LABEL_8;
    v14 = *(_QWORD *)(v13 + 80);
    v25 = *(_QWORD *)(v13 + 72);
    v26 = v14;
    v15 = sub_C93710(&v25, 32, -1) + 1;
    if ( v15 > v26 )
      v15 = v26;
    v16 = v26 - v14 + v15;
    if ( v16 > v26 )
      v16 = v26;
    if ( v16 == 6 && *(_DWORD *)v25 == 1852796476 && *(_WORD *)(v25 + 4) == 15973 )
    {
      *a3 = _mm_loadu_si128(a4);
      a3[1] = _mm_loadu_si128(a4 + 1);
    }
    else
    {
LABEL_8:
      (*(void (__fastcall **)(__int64 *))(*a1 + 144))(a1);
      v10 = _mm_loadu_si128(a3);
      v20 = a3[1].m128i_i32[0];
      v11 = a3[1].m128i_i32[1];
      v24 = v10;
      v21 = v11;
      v12 = sub_CB0A70(a1);
      if ( *(_DWORD *)(v12 + 8) == 2 )
      {
        sub_F02AE0(&v25, v12 + 32, v24.m128i_i64[0], v24.m128i_i64[1]);
        v22 = v25;
        if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __int64 *))(*a1 + 120))(
               a1,
               "File",
               1,
               0,
               &v19,
               &v25) )
        {
          sub_C2F5F0(a1, (__int64)&v22);
          (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25);
        }
      }
      else if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, int *, __int64 *))(*a1 + 120))(
                  a1,
                  "File",
                  1,
                  0,
                  &v22,
                  &v25) )
      {
        sub_C300B0(a1, (__int64)&v24);
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, int *, __int64 *))(*a1 + 120))(
             a1,
             "Line",
             1,
             0,
             &v22,
             &v25) )
      {
        sub_C2F5F0(a1, (__int64)&v20);
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, int *, __int64 *))(*a1 + 120))(
             a1,
             "Column",
             1,
             0,
             &v22,
             &v25) )
      {
        sub_C2F5F0(a1, (__int64)&v21);
        (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v25);
      }
      (*(void (__fastcall **)(__int64 *))(*a1 + 152))(a1);
    }
    return (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v23);
  }
LABEL_19:
  if ( v18 )
  {
    *a3 = _mm_loadu_si128(a4);
    a3[1] = _mm_loadu_si128(a4 + 1);
  }
  return result;
}
