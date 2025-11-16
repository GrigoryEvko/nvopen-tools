// Function: sub_1D17360
// Address: 0x1d17360
//
__int64 __fastcall sub_1D17360(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 (*v8)(void); // rdx
  __int64 v9; // rax
  __int64 (*v10)(void); // rdx
  __int64 v11; // rax
  __int64 result; // rax

  a1[4] = a2;
  a1[5] = a4;
  a1[10] = a3;
  v8 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 56LL);
  v9 = 0;
  if ( v8 != sub_1D12D20 )
  {
    v9 = v8();
    a2 = a1[4];
  }
  a1[2] = v9;
  v10 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 64LL);
  v11 = 0;
  if ( v10 != sub_1D12D30 )
  {
    v11 = v10();
    a2 = a1[4];
  }
  a1[3] = a5;
  a1[1] = v11;
  result = sub_15E0530(*(_QWORD *)a2);
  a1[8] = a6;
  a1[6] = result;
  return result;
}
