// Function: sub_818F90
// Address: 0x818f90
//
_BYTE *__fastcall sub_818F90(__int64 a1, _BYTE *a2)
{
  __int64 v2; // r15
  _BYTE *v3; // r14
  _BYTE *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rcx
  _BYTE *v7; // rbx
  __int64 v9; // rax
  __int64 v10; // rax

  v2 = 1;
  v3 = a2 - 1;
  v4 = a2;
  v5 = *(_QWORD *)(a1 + 16);
  v6 = *(_QWORD *)(a1 + 32);
  v7 = (_BYTE *)(v5 + 1);
  if ( v5 + 1 < (unsigned __int64)(v5 + v6) )
  {
    do
    {
      if ( *v7 == 10 )
      {
        if ( (unsigned int)sub_7AF220((unsigned __int64)v7) )
        {
          v9 = sub_7AF1D0((unsigned __int64)v7);
          v7 += *(_QWORD *)(v9 + 32);
          *v4 = 10;
          v10 = sub_818F90(v9, v4 + 1);
          v6 = *(_QWORD *)(a1 + 32);
          v4 = (_BYTE *)v10;
          v5 = *(_QWORD *)(a1 + 16);
        }
        else
        {
          *v4 = 10;
          ++v7;
          v5 = *(_QWORD *)(a1 + 16);
          ++v4;
          v6 = *(_QWORD *)(a1 + 32);
        }
      }
      else
      {
        ++v7;
      }
    }
    while ( (unsigned __int64)v7 < v5 + v6 );
    v2 = v4 - v3;
  }
  sub_7AED90(a1);
  *(_QWORD *)(a1 + 16) = v3;
  sub_7AED40(a1);
  *(_QWORD *)(a1 + 32) = v2;
  return v4;
}
