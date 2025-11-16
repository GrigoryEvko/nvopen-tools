// Function: sub_22AD280
// Address: 0x22ad280
//
__int64 __fastcall sub_22AD280(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  _QWORD *v8; // rdi
  __int64 v9; // r8

  v5 = sub_22AD250(a1, a2);
  if ( v5 && (v8 = (_QWORD *)sub_22AB430(v5, a3)) != 0 )
    return sub_D33D80(v8, *(_QWORD *)(a1 + 32), v6, v7, v9);
  else
    return 0;
}
