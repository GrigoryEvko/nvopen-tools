// Function: sub_17B7350
// Address: 0x17b7350
//
__int64 __fastcall sub_17B7350(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rdi

  v1 = *(unsigned int *)(a1 + 8);
  v3 = *(_QWORD *)(a1 + 8 * (3 - v1));
  if ( v3 && (sub_161E970(v3), v1 = *(unsigned int *)(a1 + 8), v4) )
  {
    v5 = *(_QWORD *)(a1 + 8 * (3 - v1));
    if ( !v5 )
      return v5;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 8 * (2 - v1));
    if ( !v5 )
      return v5;
  }
  return sub_161E970(v5);
}
