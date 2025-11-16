// Function: sub_8D5F90
// Address: 0x8d5f90
//
_BOOL8 __fastcall sub_8D5F90(__int64 a1, __int64 a2, _DWORD *a3, _QWORD *a4)
{
  __int64 v6; // r13
  __int64 v7; // r14
  _QWORD *v8; // rax
  _QWORD *v10; // rax

  *a3 = 0;
  *a4 = 0;
  v6 = sub_8D4890(a1);
  v7 = sub_8D4890(a2);
  v8 = sub_8D5CE0(v6, v7);
  *a4 = v8;
  if ( v8 )
  {
    *a3 = 1;
    return 1;
  }
  else
  {
    v10 = sub_8D5CE0(v7, v6);
    *a4 = v10;
    return v10 != 0;
  }
}
