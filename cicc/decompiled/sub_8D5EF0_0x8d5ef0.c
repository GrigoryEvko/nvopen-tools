// Function: sub_8D5EF0
// Address: 0x8d5ef0
//
_BOOL8 __fastcall sub_8D5EF0(__int64 a1, __int64 a2, _DWORD *a3, _QWORD *a4)
{
  __int64 v6; // r12
  __int64 v7; // r14
  _QWORD *v9; // rax
  _QWORD *v10; // rax

  *a3 = 0;
  *a4 = 0;
  v6 = sub_8D46C0(a1);
  v7 = sub_8D46C0(a2);
  if ( !sub_8D3A70(v6) || !sub_8D3A70(v7) )
    return 0;
  v9 = sub_8D5CE0(v6, v7);
  *a4 = v9;
  if ( v9 )
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
