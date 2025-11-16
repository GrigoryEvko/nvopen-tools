// Function: sub_2BAACB0
// Address: 0x2baacb0
//
void __fastcall sub_2BAACB0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // r9d
  __int64 v10; // [rsp+0h] [rbp-30h] BYREF
  int v11; // [rsp+8h] [rbp-28h]

  sub_2B3C700(a1);
  *(_QWORD *)(a1 + 3272) = a4;
  if ( (unsigned __int8)sub_2B0D770(a2, a3, v6, v7, v8, v9) )
  {
    v10 = 0;
    v11 = -1;
    sub_2BA65A0(a1, a2, a3, 0, &v10, 0);
  }
}
