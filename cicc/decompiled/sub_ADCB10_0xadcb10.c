// Function: sub_ADCB10
// Address: 0xadcb10
//
__int64 __fastcall sub_ADCB10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6, _BYTE *a7, int a8)
{
  int v10; // ebx
  __int64 v11; // r15
  int v12; // r10d

  v10 = (int)a7;
  if ( a7 && *a7 == 17 )
    v10 = 0;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 0;
  if ( a4 )
    v12 = sub_B9B140(v11, a3, a4);
  return sub_B05AE0(v11, 22, v12, a5, a6, v10, a2, 0, a8, 0);
}
