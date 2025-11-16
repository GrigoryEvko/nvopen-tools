// Function: sub_1623CA0
// Address: 0x1623ca0
//
__int64 __fastcall sub_1623CA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // eax
  __int64 *v4; // rdi
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 56);
  v3 = *(_DWORD *)(v2 + 8);
  if ( v3 >= *(_DWORD *)(v2 + 12) )
  {
    sub_1516630(*(_QWORD *)(a1 + 56), 0);
    v3 = *(_DWORD *)(v2 + 8);
  }
  v4 = (__int64 *)(*(_QWORD *)v2 + 8LL * v3);
  if ( v4 )
  {
    *v4 = a2;
    if ( a2 )
      sub_1623A60((__int64)v4, a2, 2);
    v3 = *(_DWORD *)(v2 + 8);
  }
  result = v3 + 1;
  *(_DWORD *)(v2 + 8) = result;
  return result;
}
