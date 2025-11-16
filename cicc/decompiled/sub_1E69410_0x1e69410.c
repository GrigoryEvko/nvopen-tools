// Function: sub_1E69410
// Address: 0x1e69410
//
__int64 __fastcall sub_1E69410(__int64 *a1, int a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v4; // r15
  __int64 v5; // rax
  __int64 v7; // rdi
  __int64 v9; // r8
  __int64 (*v10)(); // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1[3] + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a3 == v4 )
    return a3;
  v5 = *a1;
  v7 = 0;
  v9 = *(_QWORD *)(v5 + 16);
  v10 = *(__int64 (**)())(*(_QWORD *)v9 + 112LL);
  if ( v10 != sub_1D00B10 )
  {
    v15 = a3;
    v14 = ((__int64 (__fastcall *)(__int64))v10)(v9);
    a3 = v15;
    v7 = v14;
  }
  v11 = sub_1F4AF90(v7, v4, a3, 255);
  v12 = v11;
  if ( v11 && v4 != v11 )
  {
    if ( a4 > *(unsigned __int16 *)(*(_QWORD *)v11 + 20LL) )
      return 0;
    else
      sub_1E693D0((__int64)a1, a2, v11);
  }
  return v12;
}
