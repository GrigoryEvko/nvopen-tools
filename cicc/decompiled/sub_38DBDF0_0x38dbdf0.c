// Function: sub_38DBDF0
// Address: 0x38dbdf0
//
__int64 __fastcall sub_38DBDF0(__int64 a1, int a2, int a3, int a4, __int64 a5, __int64 a6, unsigned __int8 a7)
{
  int v10; // eax
  __int128 v12; // [rsp-10h] [rbp-50h]

  v10 = sub_38BE350(*(_QWORD *)(a1 + 8));
  *((_QWORD *)&v12 + 1) = a6;
  *(_QWORD *)&v12 = a5;
  return sub_39101D0(v10, a1, a2, a3, a4, a7, v12);
}
