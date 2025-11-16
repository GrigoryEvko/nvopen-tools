// Function: sub_130CC10
// Address: 0x130cc10
//
__int64 __fastcall sub_130CC10(void *a1, size_t a2)
{
  unsigned int v2; // r12d
  void *v4; // rax

  v2 = (unsigned __int8)byte_4F969C0;
  if ( !byte_4F969C0 )
  {
    v4 = mmap(a1, a2, 0, flags | 0x10u, -1, 0);
    if ( v4 != (void *)-1LL )
    {
      if ( a1 == v4 )
        return v2;
      sub_130C960(v4, a2);
    }
  }
  return 1;
}
