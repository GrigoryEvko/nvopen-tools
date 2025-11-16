// Function: sub_18713C0
// Address: 0x18713c0
//
__int64 __fastcall sub_18713C0(__int64 *a1, __int64 *a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rax
  __int64 v5; // rax
  _QWORD *v6; // r13
  void (__fastcall *v7)(__m128i *, __int64 *, __int64); // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // rcx
  __m128i v12; // xmm0
  __m128i v13; // xmm2
  unsigned __int64 v14; // r8
  __int64 v15; // r12
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  __m128i v18; // [rsp+0h] [rbp-A0h] BYREF
  void (__fastcall *v19)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-90h]
  __int64 v20; // [rsp+18h] [rbp-88h]
  __m128i v21; // [rsp+20h] [rbp-80h] BYREF
  void (__fastcall *v22)(__m128i *, __m128i *, __int64); // [rsp+30h] [rbp-70h]
  __int64 v23; // [rsp+38h] [rbp-68h]
  __m128i v24; // [rsp+40h] [rbp-60h] BYREF
  void (__fastcall *v25)(__m128i *, __m128i *, __int64); // [rsp+50h] [rbp-50h]
  __int64 v26; // [rsp+58h] [rbp-48h]
  unsigned __int64 v27; // [rsp+60h] [rbp-40h]
  __int64 v28; // [rsp+68h] [rbp-38h]
  __int64 v29; // [rsp+70h] [rbp-30h]

  v2 = 0;
  if ( (unsigned __int8)sub_1636800((__int64)a1, a2) )
    return v2;
  v4 = sub_160F9A0(a1[1], (__int64)&unk_4F98A8D, 1u);
  if ( v4 && (v5 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v4 + 104LL))(v4, &unk_4F98A8D)) != 0 )
    v6 = *(_QWORD **)(v5 + 160);
  else
    v6 = 0;
  v7 = (void (__fastcall *)(__m128i *, __int64 *, __int64))a1[22];
  v8 = v20;
  v19 = 0;
  if ( v7 )
  {
    v7(&v18, a1 + 20, 2);
    v8 = a1[23];
    v7 = (void (__fastcall *)(__m128i *, __int64 *, __int64))a1[22];
  }
  v25 = (void (__fastcall *)(__m128i *, __m128i *, __int64))v7;
  v9 = v26;
  v10 = _mm_loadu_si128(&v21);
  v11 = v23;
  v26 = v8;
  v12 = _mm_loadu_si128(&v18);
  v13 = _mm_loadu_si128(&v24);
  v23 = v9;
  v20 = v11;
  v19 = 0;
  v22 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0x1000000000LL;
  v18 = v10;
  v21 = v13;
  v24 = v12;
  v2 = sub_18708C0((__int64)&v24, (__int64)a2, v6);
  if ( HIDWORD(v28) )
  {
    v14 = v27;
    if ( (_DWORD)v28 )
    {
      v15 = 8LL * (unsigned int)v28;
      v16 = 0;
      do
      {
        v17 = *(_QWORD *)(v14 + v16);
        if ( v17 && v17 != -8 )
        {
          _libc_free(v17);
          v14 = v27;
        }
        v16 += 8;
      }
      while ( v15 != v16 );
    }
  }
  else
  {
    v14 = v27;
  }
  _libc_free(v14);
  if ( v25 )
    v25(&v24, &v24, 3);
  if ( v22 )
    v22(&v21, &v21, 3);
  if ( !v19 )
    return v2;
  v19(&v18, &v18, 3);
  return v2;
}
