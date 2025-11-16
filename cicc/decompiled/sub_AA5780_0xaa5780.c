// Function: sub_AA5780
// Address: 0xaa5780
//
__int64 __fastcall sub_AA5780(__int64 a1)
{
  _QWORD *v1; // rdi
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // r14
  int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // ebx

  v1 = (_QWORD *)(a1 + 48);
  v2 = *v1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v2 == v1 )
    return 0;
  if ( !v2 )
    BUG();
  v3 = v2 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
    return 0;
  v4 = sub_B46E30(v3);
  if ( !v4 )
    return 0;
  v5 = sub_B46EC0(v3, 0);
  if ( v4 != 1 )
  {
    v6 = 1;
    while ( v5 == sub_B46EC0(v3, v6) )
    {
      if ( v4 == ++v6 )
        return v5;
    }
    return 0;
  }
  return v5;
}
