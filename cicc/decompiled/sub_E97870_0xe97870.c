// Function: sub_E97870
// Address: 0xe97870
//
__int64 __fastcall sub_E97870(__int64 a1, __int64 a2, const void *a3, size_t a4, __int64 a5, __int64 a6, char a7)
{
  _QWORD *v10; // rax
  __int128 v12; // [rsp-10h] [rbp-50h]

  v10 = sub_E66210(*(_QWORD *)(a1 + 8), a2);
  *((_QWORD *)&v12 + 1) = a6;
  *(_QWORD *)&v12 = a5;
  return sub_E5F990((__int64)v10, a1, a2, a3, a4, a7, v12);
}
