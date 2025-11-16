// Function: sub_771390
// Address: 0x771390
//
__int64 __fastcall sub_771390(__int64 a1, int a2, unsigned int a3)
{
  __int64 v3; // rcx
  __int64 result; // rax
  int v6; // edx
  _DWORD *v7; // rsi
  unsigned int v8; // edx
  unsigned int v9; // r11d
  int *v10; // r8

  v3 = a3;
  result = a2 & (a3 + 1);
  v6 = *(_DWORD *)(a1 + 4 * result);
  while ( 1 )
  {
    v8 = a2 & v6;
    v9 = a2 & (result + 1);
    v10 = (int *)(a1 + 4LL * v9);
    if ( (v8 > (unsigned int)v3 || v8 <= (unsigned int)result && (unsigned int)v3 >= (unsigned int)result)
      && ((unsigned int)v3 >= (unsigned int)result || v8 <= (unsigned int)result) )
    {
      break;
    }
    v7 = (_DWORD *)(a1 + 4LL * (unsigned int)result);
    *(_DWORD *)(a1 + 4 * v3) = *v7;
    *v7 = 0;
    v6 = *v10;
    if ( !*v10 )
      return result;
LABEL_3:
    v3 = (unsigned int)result;
    result = v9;
  }
  v6 = *v10;
  if ( *v10 )
  {
    LODWORD(result) = v3;
    goto LABEL_3;
  }
  return result;
}
