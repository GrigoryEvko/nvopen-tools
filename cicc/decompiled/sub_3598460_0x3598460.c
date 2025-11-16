// Function: sub_3598460
// Address: 0x3598460
//
__int64 __fastcall sub_3598460(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdi
  __int64 (*v7)(); // rcx
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // rdx

  *(_QWORD *)a1 = a3;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = sub_2EA49A0(a2);
  *(_QWORD *)(a1 + 24) = sub_2EA48B0(a2);
  result = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 32) = *(_QWORD *)(*(_QWORD *)(result + 32) + 32LL);
  v6 = *(_QWORD *)(*(_QWORD *)(result + 32) + 16LL);
  v7 = *(__int64 (**)())(*(_QWORD *)v6 + 128LL);
  v8 = 0;
  if ( v7 != sub_2DAC790 )
  {
    v8 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v7)(v6, a2, 0);
    result = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  v9 = *(__int64 **)(result + 64);
  *(_QWORD *)(a1 + 64) = 0;
  v10 = *v9;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 16) = v10;
  if ( result == v10 )
  {
    result = *(_QWORD *)(*(_QWORD *)(result + 64) + 8LL);
    *(_QWORD *)(a1 + 16) = result;
  }
  return result;
}
