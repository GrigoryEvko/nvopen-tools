// Function: sub_1E816F0
// Address: 0x1e816f0
//
_QWORD *__fastcall sub_1E816F0(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  int v7; // r9d
  _QWORD *v8; // r12

  if ( *(_QWORD *)(a1 + 8LL * a2 + 616) )
    return *(_QWORD **)(a1 + 8LL * a2 + 616);
  v3 = sub_22077B0(448);
  v8 = (_QWORD *)v3;
  if ( v3 )
  {
    sub_1E81470(v3, a1, v4, v5, v6, v7);
    *v8 = off_4985780;
  }
  *(_QWORD *)(a1 + 616) = v8;
  return v8;
}
