// Function: sub_B128C0
// Address: 0xb128c0
//
__int64 __fastcall sub_B128C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v17; // [rsp+8h] [rbp-38h]

  v10 = sub_B98A20(a1, a2, a3, a4);
  v17 = sub_B98A20(a5, a2, v11, v12);
  v13 = sub_22077B0(96);
  v14 = v13;
  if ( v13 )
    sub_B12230(v13, v10, a2, a3, a4, v17, a6, a7);
  return v14;
}
