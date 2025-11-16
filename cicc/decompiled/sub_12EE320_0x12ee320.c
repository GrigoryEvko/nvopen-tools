// Function: sub_12EE320
// Address: 0x12ee320
//
void __fastcall sub_12EE320(_QWORD *a1, __int64 a2)
{
  char *v3; // rax
  char *v4; // rdi
  __int64 v5; // rdi

  v3 = (char *)(a1 + 27);
  v4 = (char *)a1[25];
  if ( v4 != v3 )
    _libc_free(v4, a2);
  v5 = a1[12];
  if ( v5 != a1[11] )
    _libc_free(v5, a2);
}
