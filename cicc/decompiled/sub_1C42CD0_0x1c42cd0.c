// Function: sub_1C42CD0
// Address: 0x1c42cd0
//
void __fastcall sub_1C42CD0(_QWORD *a1)
{
  unsigned __int64 *v2; // rbx
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // rdi
  unsigned __int64 *v5; // rbx
  unsigned __int64 *v6; // r12
  unsigned __int64 v7; // rdi
  _QWORD *v8; // rdi

  v2 = (unsigned __int64 *)a1[2];
  v3 = &v2[*((unsigned int *)a1 + 6)];
  while ( v3 != v2 )
  {
    v4 = *v2++;
    _libc_free(v4);
  }
  v5 = (unsigned __int64 *)a1[8];
  v6 = &v5[2 * *((unsigned int *)a1 + 18)];
  if ( v5 != v6 )
  {
    do
    {
      v7 = *v5;
      v5 += 2;
      _libc_free(v7);
    }
    while ( v6 != v5 );
    v6 = (unsigned __int64 *)a1[8];
  }
  if ( v6 != a1 + 10 )
    _libc_free((unsigned __int64)v6);
  v8 = (_QWORD *)a1[2];
  if ( v8 != a1 + 4 )
    _libc_free((unsigned __int64)v8);
}
