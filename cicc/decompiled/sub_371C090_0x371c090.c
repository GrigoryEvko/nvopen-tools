// Function: sub_371C090
// Address: 0x371c090
//
void __fastcall sub_371C090(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rbx
  __int64 v3; // rdx

  v1 = *(unsigned int *)(a1 + 56);
  if ( *(_DWORD *)(a1 + 56) )
  {
    v2 = 0;
    do
    {
      v3 = (unsigned int)v2++;
      sub_371C000(a1, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v3));
    }
    while ( v1 != v2 );
  }
  *(_DWORD *)(a1 + 56) = 0;
}
