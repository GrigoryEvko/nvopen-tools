// Function: sub_39D1A30
// Address: 0x39d1a30
//
__int64 __fastcall sub_39D1A30(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  char v4; // al
  _BOOL8 v5; // rcx
  size_t v6; // rdx
  void **p_s2; // rsi
  unsigned __int64 *v9; // [rsp+8h] [rbp-78h]
  char v10; // [rsp+17h] [rbp-69h] BYREF
  void **v11; // [rsp+18h] [rbp-68h] BYREF
  void *s2; // [rsp+20h] [rbp-60h] BYREF
  __int64 v13; // [rsp+28h] [rbp-58h]
  _QWORD v14[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v15[4]; // [rsp+40h] [rbp-40h] BYREF

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 144LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "id",
         1,
         0,
         &v11,
         &s2) )
  {
    sub_39D0990(a1, a2);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "class",
         1,
         0,
         &v11,
         &s2) )
  {
    sub_39D16D0(a1, a2 + 24);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  LOBYTE(v14[0]) = 0;
  v9 = (unsigned __int64 *)(a2 + 72);
  v3 = *(_QWORD *)a1;
  s2 = v14;
  v13 = 0;
  v15[0] = 0;
  v4 = (*(__int64 (__fastcall **)(__int64))(v3 + 16))(a1);
  v5 = 0;
  if ( v4 )
  {
    v6 = *(_QWORD *)(a2 + 80);
    if ( v6 == v13 )
    {
      v5 = 1;
      if ( v6 )
        v5 = memcmp(*(const void **)(a2 + 72), s2, v6) == 0;
    }
  }
  p_s2 = (void **)"preferred-register";
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, void ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "preferred-register",
         0,
         v5,
         &v10,
         &v11) )
  {
    sub_39D16D0(a1, (__int64)v9);
    p_s2 = v11;
    (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 128LL))(a1, v11);
  }
  else if ( v10 )
  {
    p_s2 = &s2;
    sub_2240AE0(v9, (unsigned __int64 *)&s2);
    *(__m128i *)(a2 + 104) = _mm_loadu_si128(v15);
  }
  if ( s2 != v14 )
  {
    p_s2 = (void **)(v14[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)s2);
  }
  return (*(__int64 (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 152LL))(a1, p_s2);
}
