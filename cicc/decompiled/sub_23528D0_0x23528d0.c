// Function: sub_23528D0
// Address: 0x23528d0
//
void __fastcall sub_23528D0(_QWORD *a1, __int64 a2)
{
  char v2; // al
  unsigned __int64 v3; // r12
  unsigned __int64 *v4; // rbx

  v2 = *((_BYTE *)a1 + 16);
  if ( (v2 & 2) != 0 )
    sub_2352860(a1, a2);
  v3 = *a1;
  if ( (v2 & 1) != 0 )
  {
    if ( v3 )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v3 + 8LL))(*a1);
  }
  else
  {
    v4 = (unsigned __int64 *)(v3 + 32LL * *((unsigned int *)a1 + 2));
    if ( v4 != (unsigned __int64 *)v3 )
    {
      do
      {
        v4 -= 4;
        if ( (unsigned __int64 *)*v4 != v4 + 2 )
          j_j___libc_free_0(*v4);
      }
      while ( v4 != (unsigned __int64 *)v3 );
      v3 = *a1;
    }
    if ( (_QWORD *)v3 != a1 + 2 )
      _libc_free(v3);
  }
}
