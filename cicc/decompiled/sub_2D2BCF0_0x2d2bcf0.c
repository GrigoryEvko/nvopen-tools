// Function: sub_2D2BCF0
// Address: 0x2d2bcf0
//
unsigned __int64 __fastcall sub_2D2BCF0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned int *i; // rax

  v6 = *(_QWORD *)a1;
  v7 = *(unsigned int *)(*(_QWORD *)a1 + 192LL);
  if ( (_DWORD)v7 )
    return sub_2D2BC70(a1, a2, v7, a4, a5, a6);
  v8 = *(unsigned int *)(v6 + 196);
  if ( (_DWORD)v8 )
  {
    for ( i = (unsigned int *)(v6 + 4); a2 >= *i; i += 2 )
    {
      v7 = (unsigned int)(v7 + 1);
      if ( (_DWORD)v8 == (_DWORD)v7 )
        return sub_2D29C80(a1, v8, v7, a4, v8, a6);
    }
    v8 = (unsigned int)v7;
  }
  return sub_2D29C80(a1, v8, v7, a4, v8, a6);
}
