// Function: sub_FCD210
// Address: 0xfcd210
//
__int64 __fastcall sub_FCD210(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r12

  v6 = *a1;
  v7 = sub_FC95E0(*a1, a2, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
    v12 = (__int64)v7;
  else
    v12 = sub_FCBA10(v6, a2, v8, v9, v10, v11);
  sub_FCCAD0(v6, a2, v8, v9, v10, v11);
  return v12;
}
