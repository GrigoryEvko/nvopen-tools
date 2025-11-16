// Function: sub_1E62D10
// Address: 0x1e62d10
//
__int64 __fastcall sub_1E62D10(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 *v2; // r14
  __int64 *v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r12

  v1 = a1[4];
  if ( !v1 )
    return 0;
  v2 = *(__int64 **)(v1 + 72);
  v3 = *(__int64 **)(v1 + 64);
  if ( v3 == v2 )
    return 0;
  v4 = 0;
  do
  {
    v5 = *v3;
    if ( (unsigned __int8)sub_1E62AD0(a1, *v3) )
    {
      if ( v4 )
        return 0;
      v4 = v5;
    }
    ++v3;
  }
  while ( v2 != v3 );
  return v4;
}
