// Function: sub_AE44F0
// Address: 0xae44f0
//
__int64 __fastcall sub_AE44F0(__int64 a1)
{
  unsigned __int8 *v1; // rdx
  unsigned __int8 *v2; // rcx
  unsigned __int8 *v3; // rax

  v1 = *(unsigned __int8 **)(a1 + 32);
  v2 = &v1[*(_QWORD *)(a1 + 40)];
  if ( v1 == v2 )
    return 0;
  v3 = v1 + 1;
  if ( v1 + 1 == v2 )
    return *v1;
  do
  {
    if ( *v1 < *v3 )
      v1 = v3;
    ++v3;
  }
  while ( v2 != v3 );
  if ( v2 == v1 )
    return 0;
  else
    return *v1;
}
