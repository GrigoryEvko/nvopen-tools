// Function: sub_818EC0
// Address: 0x818ec0
//
unsigned __int64 __fastcall sub_818EC0(int a1, char *a2, __int64 a3, __int64 a4, _QWORD *a5, _DWORD *a6)
{
  char v6; // r11
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rcx
  char v12; // dl

  v6 = *a2;
  *a6 = 0;
  result = (unsigned int)a2[4];
  if ( (_DWORD)result == 3 || (_DWORD)result == 8 )
  {
    result = (unsigned __int64)(unsigned __int8)a2[7] << 16;
    v10 = (unsigned __int8)a2[5] | result | ((unsigned __int64)(unsigned __int8)a2[6] << 8);
    if ( v10 <= 0x32 )
    {
      v11 = *(_QWORD **)(a4 + 8 * v10 - 8);
      if ( v10 != a3 )
        return result;
    }
    else
    {
      v11 = *(_QWORD **)(a4 + 392);
      for ( result = 50; result != v10; ++result )
        v11 = (_QWORD *)*v11;
      if ( v10 != a3 )
        return result;
    }
    if ( !v11[1] && a1 == 1 )
    {
      v12 = *(a2 - 1);
      result = (unsigned __int64)(a2 - 1);
      if ( v12 == 32 || v12 == 9 )
      {
        do
        {
          do
            v12 = *(_BYTE *)--result;
          while ( v12 == 9 );
        }
        while ( v12 == 32 );
      }
      if ( v6 == 7 && !*(_BYTE *)(result - 1) && v12 == 4 )
        result -= 2LL;
      if ( *(_BYTE *)result == 44 )
        *a5 -= &a2[-result];
    }
  }
  return result;
}
