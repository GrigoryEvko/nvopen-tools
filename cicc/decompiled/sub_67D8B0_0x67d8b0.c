// Function: sub_67D8B0
// Address: 0x67d8b0
//
__int64 __fastcall sub_67D8B0(__int64 a1, char a2, int a3)
{
  _DWORD *v4; // rax
  unsigned int v5; // r8d
  _QWORD key[4]; // [rsp+0h] [rbp-20h] BYREF

  key[0] = a1;
  v4 = bsearch(key, &off_4A44400, 0xD2Cu, 0x10u, (__compar_fn_t)sub_67BDB0);
  v5 = 1;
  if ( v4 )
  {
    sub_67D850(v4[2], a2, a3);
    return 0;
  }
  return v5;
}
