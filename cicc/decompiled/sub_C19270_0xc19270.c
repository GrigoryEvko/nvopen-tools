// Function: sub_C19270
// Address: 0xc19270
//
void __fastcall sub_C19270(__int64 a1, __int64 a2)
{
  char **v2; // r14
  __int64 v3; // rdi
  char *v4; // [rsp+10h] [rbp-80h] BYREF
  char *v5[2]; // [rsp+18h] [rbp-78h] BYREF
  _BYTE v6[104]; // [rsp+28h] [rbp-68h] BYREF

  v2 = (char **)(a1 + 8);
  v4 = *(char **)a1;
  v5[0] = v6;
  v5[1] = (char *)0xC00000000LL;
  if ( *(_DWORD *)(a1 + 16) )
    sub_C15E20((__int64)v5, (char **)(a1 + 8));
  while ( sub_C185F0(a2, (__int64)&v4, (__int64)(v2 - 10)) )
  {
    v3 = (__int64)v2;
    *(v2 - 1) = *(v2 - 10);
    v2 -= 9;
    sub_C15E20(v3, v2);
  }
  *(v2 - 1) = v4;
  sub_C15E20((__int64)v2, v5);
  if ( v5[0] != v6 )
    _libc_free(v5[0], v5);
}
