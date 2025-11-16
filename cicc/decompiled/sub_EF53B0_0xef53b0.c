// Function: sub_EF53B0
// Address: 0xef53b0
//
__int64 __fastcall sub_EF53B0(_QWORD *a1, char *a2, _BYTE *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r13
  __int64 v9; // rdx
  _WORD *v10; // rax
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r14
  unsigned __int64 v13; // r8
  __int64 v15; // rdi
  __int64 v16; // rax

  v6 = (__int64 *)a2;
  v9 = a1[1];
  v10 = (_WORD *)*a1;
  v11 = v9 - *a1;
  if ( v11 > 1 && *v10 == 29779 )
  {
    v15 = (__int64)(a1 + 101);
    a2 = "std";
    *(_QWORD *)(v15 - 808) = v10 + 1;
    v12 = sub_EE68C0(v15, "std");
    if ( !v12 )
      return 0;
    v9 = a1[1];
    v10 = (_WORD *)*a1;
  }
  else
  {
    v12 = 0;
  }
  v13 = 0;
  if ( v10 == (_WORD *)v9 || *(_BYTE *)v10 != 83 )
    return sub_EF4820((__int64)a1, v6, v12, v13);
  v16 = sub_EE9AE0((__int64)a1, (__int64)a2, v9, v11, 0, a6);
  v13 = v16;
  if ( v16 )
  {
    if ( *(_BYTE *)(v16 + 8) == 27 )
      return sub_EF4820((__int64)a1, v6, v12, v13);
    if ( !v12 )
    {
      *a3 = 1;
      return v13;
    }
  }
  return 0;
}
