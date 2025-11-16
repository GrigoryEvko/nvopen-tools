// Function: sub_356E200
// Address: 0x356e200
//
__int64 __fastcall sub_356E200(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 result; // rax

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = a3;
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  a1[4] = v4;
  result = *(_QWORD *)(a2 + 32);
  a1[5] = result;
  return result;
}
