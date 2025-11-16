// Function: sub_AC3090
// Address: 0xac3090
//
__int64 __fastcall sub_AC3090(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v7; // rax
  __int64 v8; // rdi

  v5 = *(_QWORD *)(a1 + 24);
  if ( v5 != *a2 )
    return 0;
  v7 = sub_C33340(a1, a2, a3, a4, a5);
  v8 = a1 + 24;
  if ( v5 == v7 )
    return sub_C3E590(v8);
  else
    return sub_C33D00(v8);
}
