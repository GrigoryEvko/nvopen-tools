// Function: sub_D49F60
// Address: 0xd49f60
//
unsigned __int64 __fastcall sub_D49F60(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  unsigned int v4; // r12d
  unsigned __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 40) != v2 )
  {
    v3 = 0;
    v4 = 0;
    do
    {
      sub_D49BF0(*(_QWORD *)(v2 + 8 * v3), a2, 0, 1, 0);
      v2 = *(_QWORD *)(a1 + 32);
      v3 = ++v4;
      result = (*(_QWORD *)(a1 + 40) - v2) >> 3;
    }
    while ( v4 < result );
  }
  return result;
}
