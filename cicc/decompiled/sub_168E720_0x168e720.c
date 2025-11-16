// Function: sub_168E720
// Address: 0x168e720
//
__int64 *__fastcall sub_168E720(__int64 a1)
{
  __int64 *v1; // r8
  __int64 *v2; // rax
  __int64 v3; // rdx

  v1 = *(__int64 **)(a1 + 272);
  if ( *(_DWORD *)(a1 + 280) && (*v1 == -8 || !*v1) )
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
