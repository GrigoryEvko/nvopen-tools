// Function: sub_1594470
// Address: 0x1594470
//
__int64 __fastcall sub_1594470(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // r13

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 1856);
  if ( !v2 )
  {
    v2 = sub_1648A60(24, 0);
    if ( v2 )
    {
      v4 = sub_16432D0(a1);
      sub_1648CB0(v2, v4, 16);
      *(_DWORD *)(v2 + 20) &= 0xF0000000;
    }
    v5 = *(_QWORD *)(v1 + 1856);
    *(_QWORD *)(v1 + 1856) = v2;
    if ( v5 )
    {
      sub_164BE60(v5);
      sub_1648B90(v5);
      return *(_QWORD *)(v1 + 1856);
    }
  }
  return v2;
}
