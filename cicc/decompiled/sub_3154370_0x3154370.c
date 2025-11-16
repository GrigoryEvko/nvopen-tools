// Function: sub_3154370
// Address: 0x3154370
//
__int64 __fastcall sub_3154370(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi
  __int64 v3; // r12
  __int64 v4; // rax

  v1 = a1[2];
  v2 = *a1;
  v3 = v2;
  if ( v1 == v2 )
    return v1;
  while ( 1 )
  {
    v4 = sub_220EF30(v2);
    v2 = v4;
    if ( v4 == v1 )
      break;
    while ( *(_DWORD *)(v3 + 32) < *(_DWORD *)(v4 + 32) )
    {
      v3 = v2;
      v4 = sub_220EF30(v2);
      v2 = v4;
      if ( v4 == v1 )
        return v3;
    }
  }
  return v3;
}
