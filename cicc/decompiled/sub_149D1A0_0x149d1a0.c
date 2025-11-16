// Function: sub_149D1A0
// Address: 0x149d1a0
//
void *__fastcall sub_149D1A0(__int64 a1, _BYTE *a2, __int64 a3, int a4)
{
  _BYTE *v5; // rax
  size_t v6; // rdx
  __int64 v8; // rax
  __int64 v9; // r14
  void *v10; // r15
  size_t v11; // r13
  __int64 v12; // rbx
  void *s2; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]

  v5 = sub_149A7F0(a2, a3);
  n = v6;
  s2 = v5;
  if ( !v6 )
    return s2;
  v8 = sub_149D080(
         *(_QWORD *)(a1 + 152),
         *(_QWORD *)(a1 + 160),
         &s2,
         (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))sub_149A790);
  v9 = *(_QWORD *)(a1 + 160);
  if ( v8 != v9 )
  {
    v10 = s2;
    v11 = n;
    v12 = v8;
    while ( v11 == *(_QWORD *)(v12 + 8) && (!v11 || !memcmp(*(const void **)v12, v10, v11)) )
    {
      if ( *(_DWORD *)(v12 + 32) == a4 )
        return *(void **)(v12 + 16);
      v12 += 40;
      if ( v9 == v12 )
        return 0;
    }
  }
  return 0;
}
