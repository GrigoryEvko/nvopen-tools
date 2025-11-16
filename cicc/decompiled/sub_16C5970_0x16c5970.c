// Function: sub_16C5970
// Address: 0x16c5970
//
__int64 __fastcall sub_16C5970(size_t *a1, int a2, __off_t a3, int a4)
{
  bool v6; // cf
  int v7; // ecx
  int v8; // edx
  size_t v9; // rsi
  void *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8

  v6 = a4 == 0;
  if ( a4 == 1 )
  {
    v7 = 1;
    v8 = 3;
  }
  else
  {
    v7 = 2;
    v8 = v6 ? 1 : 3;
  }
  v9 = *a1;
  v10 = mmap(0, *a1, v8, v7, a2, a3);
  a1[1] = (size_t)v10;
  if ( v10 == (void *)-1LL )
  {
    sub_2241E50(0, v9, v11, v12, v13);
    return (unsigned int)*__errno_location();
  }
  else
  {
    sub_2241E40(0, v9, v11, v12, v13);
    return 0;
  }
}
