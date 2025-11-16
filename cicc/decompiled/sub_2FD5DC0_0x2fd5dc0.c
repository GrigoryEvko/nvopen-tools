// Function: sub_2FD5DC0
// Address: 0x2fd5dc0
//
__int64 __fastcall sub_2FD5DC0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        char a7,
        int a8)
{
  __int64 (*v11)(void); // rdx
  __int64 v12; // rax
  __int64 v13; // rax

  *(_QWORD *)(a1 + 32) = a2;
  v11 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v12 = 0;
  if ( v11 != sub_2DAC790 )
  {
    v12 = v11();
    a2 = *(_QWORD *)(a1 + 32);
  }
  *(_QWORD *)a1 = v12;
  *(_QWORD *)(a1 + 8) = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 32LL);
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 40) = a5;
  *(_QWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 48) = a6;
  *(_DWORD *)(a1 + 60) = a8;
  *(_BYTE *)(a1 + 57) = a7;
  *(_BYTE *)(a1 + 56) = a3;
  return a3;
}
