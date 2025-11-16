// Function: sub_B190C0
// Address: 0xb190c0
//
__int64 __fastcall sub_B190C0(_QWORD *a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // r13
  int v3; // r12d
  __int64 v4; // r14
  int v5; // r15d
  unsigned int v6; // ebx

  v1 = *(_QWORD *)(*a1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v1 == *a1 + 48LL )
    return 1;
  if ( !v1 )
    BUG();
  v2 = v1 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v1 - 24) - 30 > 0xA )
    return 1;
  v3 = sub_B46E30(v2);
  if ( !v3 )
    return 1;
  v4 = a1[1];
  v5 = 0;
  v6 = 0;
  while ( 1 )
  {
    while ( v4 != sub_B46EC0(v2, v6) )
    {
      if ( v3 == ++v6 )
        return 1;
    }
    if ( v5 == 1 )
      break;
    ++v6;
    v5 = 1;
    if ( v3 == v6 )
      return 1;
  }
  return 0;
}
