// Function: sub_2B397C0
// Address: 0x2b397c0
//
bool __fastcall sub_2B397C0(unsigned int ***a1, _DWORD *a2)
{
  unsigned int *v2; // rax
  unsigned int *v3; // rdi
  __int64 v4; // rcx
  __int64 v5; // r8

  v2 = **a1;
  v3 = &v2[(_QWORD)(*a1)[1]];
  if ( v2 != v3 )
  {
    v4 = 0;
    do
    {
      v5 = *v2;
      if ( v5 != v4 && (_DWORD)v5 != *a2 )
        break;
      ++v2;
      ++v4;
    }
    while ( v3 != v2 );
  }
  return v3 == v2;
}
