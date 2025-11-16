// Function: sub_38ABC90
// Address: 0x38abc90
//
__int64 __fastcall sub_38ABC90(__int64 **a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v6; // r12d
  __int64 v8; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // r13
  __int64 v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_38AB270(a1, v11, a3, a4, a5, a6);
  if ( (_BYTE)v6 )
    return v6;
  v8 = v11[0];
  v9 = sub_1648A60(56, 1u);
  v10 = v9;
  if ( v9 )
    sub_15F7290((__int64)v9, v8, 0);
  *a2 = v10;
  return v6;
}
