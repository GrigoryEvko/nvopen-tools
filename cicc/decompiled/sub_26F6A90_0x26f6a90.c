// Function: sub_26F6A90
// Address: 0x26f6a90
//
void __fastcall sub_26F6A90(_QWORD *a1)
{
  _QWORD *v1; // r13
  unsigned __int64 v2; // r14
  _QWORD *v3; // rbx
  _QWORD *v4; // r12

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = (unsigned __int64)v1;
      sub_26F6A90(v1[3]);
      v3 = (_QWORD *)v1[8];
      v4 = (_QWORD *)v1[7];
      v1 = (_QWORD *)v1[2];
      if ( v3 != v4 )
      {
        do
        {
          if ( *v4 )
            (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v4 + 8LL))(*v4);
          ++v4;
        }
        while ( v3 != v4 );
        v4 = *(_QWORD **)(v2 + 56);
      }
      if ( v4 )
        j_j___libc_free_0((unsigned __int64)v4);
      j_j___libc_free_0(v2);
    }
    while ( v1 );
  }
}
