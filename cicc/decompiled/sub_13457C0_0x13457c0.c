// Function: sub_13457C0
// Address: 0x13457c0
//
__int64 __fastcall sub_13457C0(_BYTE *a1, __int64 a2, __int64 *a3, char a4, char a5)
{
  unsigned int v6; // r13d
  unsigned int v8; // eax
  void *v9; // rdi
  size_t v10; // rdx
  __int64 *v11; // [rsp+0h] [rbp-20h]

  if ( a4 && (*a3 & 0x2000) == 0 )
  {
    v11 = a3;
    v8 = sub_1343140(a1, (unsigned int *)a2, a3, 0, a3[2] & 0xFFFFFFFFFFFFF000LL);
    a3 = v11;
    v6 = v8;
    if ( (_BYTE)v8 )
      return v6;
    v6 = 0;
    if ( !a5 )
      return v6;
  }
  else
  {
    v6 = 0;
    if ( !a5 )
      return v6;
  }
  if ( (*a3 & 0x8000) != 0 )
    return v6;
  v9 = (void *)(a3[1] & 0xFFFFFFFFFFFFF000LL);
  v10 = a3[2] & 0xFFFFFFFFFFFFF000LL;
  if ( *(__int64 (__fastcall ***)(int, int, int, int, int, int, int))(a2 + 8) == &off_49E8020 )
  {
    sub_1341200(v9, v10);
    return v6;
  }
  memset(v9, 0, v10);
  return 0;
}
