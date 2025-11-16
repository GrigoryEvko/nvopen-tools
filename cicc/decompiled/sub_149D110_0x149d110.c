// Function: sub_149D110
// Address: 0x149d110
//
__int64 __fastcall sub_149D110(__int64 a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // rax
  size_t v4; // rdx
  unsigned int v5; // r8d
  __int64 v7; // rax
  void *s2; // [rsp+0h] [rbp-20h] BYREF
  size_t n; // [rsp+8h] [rbp-18h]

  v3 = sub_149A7F0(a2, a3);
  n = v4;
  s2 = v3;
  if ( v4
    && (v7 = sub_149D080(
               *(_QWORD *)(a1 + 152),
               *(_QWORD *)(a1 + 160),
               &s2,
               (unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD))sub_149A790),
        v7 != *(_QWORD *)(a1 + 160))
    && *(_QWORD *)(v7 + 8) == n )
  {
    v5 = 1;
    if ( n )
      LOBYTE(v5) = memcmp(*(const void **)v7, s2, n) == 0;
  }
  else
  {
    return 0;
  }
  return v5;
}
