// Function: sub_BD3960
// Address: 0xbd3960
//
__int64 __fastcall sub_BD3960(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // edx

  v1 = *(_QWORD *)(a1 + 16);
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
