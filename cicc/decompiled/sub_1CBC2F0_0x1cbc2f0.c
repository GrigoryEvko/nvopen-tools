// Function: sub_1CBC2F0
// Address: 0x1cbc2f0
//
__int64 __fastcall sub_1CBC2F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rbx
  __int64 v4; // r13
  __int64 result; // rax
  _QWORD *v6; // r12
  __int64 v7; // rax

  v2 = *(_QWORD *)(a2 + 80);
  v3 = *(_QWORD *)(a1 + 280);
  v4 = *(_QWORD *)(a1 + 272);
  result = v2 - 24;
  if ( v2 )
    v2 -= 24;
  for ( ; v4 != v3; v3 -= 8 )
  {
    v6 = (_QWORD *)sub_1CBC2E0(*(_QWORD *)(v3 - 8));
    result = sub_15F8F00((__int64)v6);
    if ( (_BYTE)result )
    {
      v7 = sub_157ED20(v2);
      result = sub_15F22F0(v6, v7);
    }
  }
  return result;
}
