// Function: sub_E71E80
// Address: 0xe71e80
//
__int64 __fastcall sub_E71E80(__int64 *a1, __int64 a2, __int64 a3, char a4)
{
  char v5; // r13
  _BYTE *v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8

  v5 = a3;
  v6 = *(_BYTE **)(a1[1] + 152);
  (*(void (__fastcall **)(_BYTE *, __int64, __int64, __int64 *))(*(_QWORD *)v6 + 32LL))(v6, a2, a3, a1);
  v7 = (unsigned int)sub_E71E20(a1[1], v5);
  if ( v6[349] && a4 )
    return sub_E71DA0(a1, v9, v7, v8, v9);
  else
    return sub_E9A5B0(a1, v9, v7, 0);
}
