// Function: sub_3069010
// Address: 0x3069010
//
bool __fastcall sub_3069010(
        unsigned int ***a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        _BYTE *a8,
        _DWORD *a9,
        unsigned int *a10)
{
  unsigned int *v10; // rax
  unsigned int *v11; // rdi
  __int64 v12; // rsi
  bool v13; // dl
  unsigned int v14; // edx

  v10 = **a1;
  v11 = &v10[(_QWORD)(*a1)[1]];
  if ( v10 != v11 )
  {
    v12 = 0;
    do
    {
      v14 = *v10;
      if ( *v10 == -1 )
      {
        if ( *(_QWORD *)(a7 + 8) - 1LL == v12 && !*a8 )
          return v11 == v10;
      }
      else
      {
        if ( v14 >= 2 * *a9 )
          return v11 == v10;
        if ( *a10 == -1 )
        {
          *a10 = v14;
          v13 = *(_QWORD *)(a7 + 8) - 1LL != v12;
        }
        else
        {
          *a8 = 1;
          v13 = *a10 == *v10;
        }
        if ( !v13 )
          return v11 == v10;
      }
      ++v10;
      ++v12;
    }
    while ( v11 != v10 );
  }
  return v11 == v10;
}
