// Function: sub_1648EF0
// Address: 0x1648ef0
//
__int64 __fastcall sub_1648EF0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // edx

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 0;
  v2 = 0;
  do
  {
    v1 = *(_QWORD *)(v1 + 8);
    ++v2;
  }
  while ( v1 );
  return v2;
}
