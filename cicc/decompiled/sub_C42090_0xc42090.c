// Function: sub_C42090
// Address: 0xc42090
//
unsigned __int64 __fastcall sub_C42090(__int64 *a1)
{
  __int64 v2; // rdi
  unsigned __int64 v3; // rax
  void **v4; // rdi
  unsigned __int64 v6; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v7; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1[1];
  if ( !v2 )
    return sub_C41F00(a1);
  v3 = sub_C42050((void **)(v2 + 24));
  v4 = (void **)a1[1];
  v7 = v3;
  v6 = sub_C42050(v4);
  return sub_C41E80((__int64 *)&v6, (__int64 *)&v7);
}
