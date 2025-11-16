// Function: sub_C14A60
// Address: 0xc14a60
//
__int64 *__fastcall sub_C14A60(__int64 a1)
{
  __int64 *v1; // r8
  __int64 *v2; // rax
  __int64 v3; // rdx

  v1 = *(__int64 **)(a1 + 304);
  if ( *(_DWORD *)(a1 + 312) && (*v1 == -8 || !*v1) )
  {
    v2 = v1 + 1;
    do
    {
      do
      {
        v3 = *v2;
        v1 = v2++;
      }
      while ( v3 == -8 );
    }
    while ( !v3 );
  }
  return v1;
}
