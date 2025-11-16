// Function: sub_130C960
// Address: 0x130c960
//
int __fastcall sub_130C960(void *a1, size_t a2)
{
  _BYTE *v2; // rax
  int *v3; // rax
  char v5[72]; // [rsp+0h] [rbp-50h] BYREF

  LODWORD(v2) = munmap(a1, a2);
  if ( (_DWORD)v2 == -1 )
  {
    v3 = __errno_location();
    sub_130AA70(*v3, v5, 0x40u);
    sub_130ACF0("<jemalloc>: Error in munmap(): %s\n", v5);
    v2 = byte_4F969A5;
    if ( byte_4F969A5[0] )
      abort();
  }
  return (int)v2;
}
