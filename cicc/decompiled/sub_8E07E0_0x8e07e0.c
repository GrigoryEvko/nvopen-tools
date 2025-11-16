// Function: sub_8E07E0
// Address: 0x8e07e0
//
__int64 __fastcall sub_8E07E0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v5; // [rsp-38h] [rbp-38h] BYREF

  if ( a1 == a2 )
    return 1;
  v2 = sub_72D2E0(a1);
  v3 = sub_72D2E0(a2);
  return sub_8DFA20(v3, 0, 0, 0, 0, v2, 0, 0, 0, (__int64)&v5, 0);
}
