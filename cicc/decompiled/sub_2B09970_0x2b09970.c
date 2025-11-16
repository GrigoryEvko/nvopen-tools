// Function: sub_2B09970
// Address: 0x2b09970
//
bool __fastcall sub_2B09970(unsigned int ***a1, unsigned int *a2)
{
  unsigned int *v3; // rax
  unsigned int *v4; // rsi
  __int64 v5; // rdi
  __int64 i; // rcx
  __int64 v7; // rdx

  v3 = **a1;
  v4 = &v3[(_QWORD)(*a1)[1]];
  if ( v3 != v4 )
  {
    v5 = *a2;
    for ( i = v5 - 1; ; --i )
    {
      v7 = *v3;
      if ( (_DWORD)v7 != (_DWORD)v5 && i != v7 )
        break;
      if ( v4 == ++v3 )
        return 1;
    }
  }
  return v4 == v3;
}
