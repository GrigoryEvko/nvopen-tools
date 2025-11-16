// Function: sub_B4FCC0
// Address: 0xb4fcc0
//
__int64 __fastcall sub_B4FCC0(__int64 a1, unsigned __int64 a2, int a3, _DWORD *a4)
{
  int v8; // r9d
  int v9; // ecx
  unsigned int v10; // edx
  __int64 v11; // rax
  int v12; // eax

  if ( !a3 )
    return 0;
  v8 = 0;
  while ( a2 )
  {
    v9 = v8;
    v10 = 0;
    v11 = 0;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 4 * v11);
      if ( v12 >= 0 && v12 != v9 )
        break;
      v9 += a3;
      v11 = ++v10;
      if ( v10 >= a2 )
      {
        if ( v10 == a2 )
          goto LABEL_11;
        break;
      }
    }
    if ( a3 == ++v8 )
      return 0;
  }
LABEL_11:
  *a4 = v8;
  return 1;
}
