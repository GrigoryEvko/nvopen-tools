// Function: sub_1F15220
// Address: 0x1f15220
//
_QWORD *__fastcall sub_1F15220(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 (*v11)(); // rcx
  __int64 v12; // rdx
  __int64 (*v13)(void); // rdx
  __int64 v14; // rax
  _QWORD *result; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // rdx

  v7 = *(_QWORD *)(a5 + 256);
  *a1 = a2;
  a1[1] = a3;
  a1[2] = a4;
  a1[3] = a5;
  v9 = *(_QWORD *)(v7 + 40);
  a1[5] = a6;
  a1[4] = v9;
  v10 = *(_QWORD *)(v7 + 16);
  v11 = *(__int64 (**)())(*(_QWORD *)v10 + 40LL);
  v12 = 0;
  if ( v11 != sub_1D00B00 )
  {
    v12 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v11)(v10, a2, 0);
    v7 = *(_QWORD *)(a5 + 256);
  }
  a1[6] = v12;
  v13 = *(__int64 (**)(void))(**(_QWORD **)(v7 + 16) + 112LL);
  v14 = 0;
  if ( v13 != sub_1D00B10 )
    v14 = v13();
  a1[7] = v14;
  a1[9] = 0;
  a1[8] = a7;
  a1[14] = a1 + 16;
  a1[15] = 0x400000000LL;
  a1[20] = a1 + 22;
  a1[49] = a1 + 11;
  a1[10] = 0;
  a1[11] = 0;
  a1[12] = 0;
  a1[13] = 0;
  a1[21] = 0;
  a1[22] = 0;
  a1[23] = 1;
  memset(a1 + 25, 0, 0xB8u);
  result = a1 + 54;
  a1[48] = 0;
  v16 = a1 + 220;
  *(v16 - 170) = 0;
  *(v16 - 169) = 0;
  *(v16 - 168) = 0;
  *((_DWORD *)v16 - 334) = 0;
  do
  {
    *result = 0;
    result[12] = result + 14;
    v17 = result + 19;
    result += 83;
    *(result - 82) = 0;
    *(result - 81) = 0;
    *(result - 80) = 0;
    *(result - 79) = 0;
    *(result - 78) = 0;
    *(result - 77) = 0;
    *((_DWORD *)result - 152) = 0;
    *(result - 75) = 0;
    *((_DWORD *)result - 144) = 0;
    *(result - 74) = 0;
    *((_DWORD *)result - 146) = 0;
    *((_DWORD *)result - 145) = 0;
    *((_DWORD *)result - 140) = 0;
    *((_DWORD *)result - 139) = 0;
    *(result - 69) = 0;
    *(result - 68) = 0;
    *(result - 66) = v17;
    *((_DWORD *)result - 130) = 0;
    *((_DWORD *)result - 129) = 16;
  }
  while ( result != v16 );
  return result;
}
