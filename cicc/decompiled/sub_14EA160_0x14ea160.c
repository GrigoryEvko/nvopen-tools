// Function: sub_14EA160
// Address: 0x14ea160
//
__int64 __fastcall sub_14EA160(_QWORD *a1)
{
  _QWORD *v1; // r13
  _QWORD *v2; // r14
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 result; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      v2 = v1;
      sub_14EA160(v1[3]);
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
        v4 = (_QWORD *)v2[7];
      }
      if ( v4 )
        j_j___libc_free_0(v4, v2[9] - (_QWORD)v4);
      result = j_j___libc_free_0(v2, 80);
    }
    while ( v1 );
  }
  return result;
}
