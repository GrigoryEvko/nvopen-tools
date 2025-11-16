// Function: sub_2FB1ED0
// Address: 0x2fb1ed0
//
_QWORD *__fastcall sub_2FB1ED0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 (*v13)(); // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *result; // rax
  _QWORD *v18; // rbx
  _QWORD *v19; // rdx

  v10 = *(_QWORD *)(a4 + 24);
  *a1 = a2;
  a1[1] = a3;
  a1[2] = a4;
  v11 = *(_QWORD *)(v10 + 32);
  a1[4] = a5;
  a1[3] = v11;
  v12 = *(_QWORD *)(v10 + 16);
  v13 = *(__int64 (**)())(*(_QWORD *)v12 + 128LL);
  v14 = 0;
  if ( v13 != sub_2DAC790 )
  {
    v14 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v13)(v12, a2, 0);
    v10 = *(_QWORD *)(a4 + 24);
  }
  a1[5] = v14;
  v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v10 + 16) + 200LL))(*(_QWORD *)(v10 + 16));
  a1[7] = a6;
  a1[6] = v15;
  a1[9] = 0;
  a1[8] = a7;
  a1[14] = a1 + 16;
  a1[15] = 0x400000000LL;
  a1[20] = a1 + 22;
  a1[48] = a1 + 11;
  a1[10] = 0;
  a1[11] = 0;
  a1[12] = 0;
  a1[13] = 0;
  a1[21] = 0;
  a1[22] = 0;
  a1[23] = 1;
  memset(a1 + 24, 0, 0xB8u);
  v16 = a1 + 24;
  a1[47] = 0;
  do
  {
    *v16 = 0;
    v16 += 2;
    *(v16 - 1) = 0;
  }
  while ( v16 != a1 + 42 );
  a1[49] = 0;
  result = a1 + 53;
  v18 = a1 + 231;
  *(v18 - 181) = 0;
  *(v18 - 180) = 0;
  *((_DWORD *)v18 - 358) = 0;
  do
  {
    *result = 0;
    result[5] = result + 7;
    result[18] = result + 20;
    v19 = result + 25;
    result += 89;
    *(result - 88) = 0;
    *(result - 87) = 0;
    *(result - 86) = 0;
    *(result - 85) = 0;
    *((_DWORD *)result - 166) = 0;
    *((_DWORD *)result - 165) = 6;
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
    *(result - 66) = v19;
    *((_DWORD *)result - 130) = 0;
    *((_DWORD *)result - 129) = 16;
  }
  while ( result != v18 );
  return result;
}
