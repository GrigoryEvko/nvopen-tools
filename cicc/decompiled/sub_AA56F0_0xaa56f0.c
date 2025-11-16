// Function: sub_AA56F0
// Address: 0xaa56f0
//
__int64 __fastcall sub_AA56F0(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // r12
  int v3; // ebx
  __int64 result; // rax

  v1 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == a1 + 48 )
    return 0;
  if ( !v1 )
    BUG();
  v2 = v1 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 > 0xA )
    return 0;
  v3 = sub_B46E30(v2);
  if ( !v3 )
    return 0;
  result = sub_B46EC0(v2, 0);
  if ( v3 != 1 )
    return 0;
  return result;
}
