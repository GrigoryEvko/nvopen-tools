// Function: sub_35E6C40
// Address: 0x35e6c40
//
__int64 __fastcall sub_35E6C40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rdi
  __int64 (*v7)(void); // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 result; // rax
  __int64 v11; // rdi

  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)a1 = &unk_4A3ADC8;
  v4 = *(_QWORD *)(a2 + 8);
  *(_QWORD *)(a1 + 16) = v4;
  v5 = **(_QWORD **)(a3 + 32);
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 24) = v5;
  v6 = *(_QWORD *)(v4 + 16);
  *(_QWORD *)(a1 + 40) = v6;
  v7 = *(__int64 (**)(void))(*(_QWORD *)v6 + 128LL);
  v8 = 0;
  if ( v7 != sub_2DAC790 )
  {
    v8 = v7();
    v6 = *(_QWORD *)(a1 + 40);
  }
  *(_QWORD *)(a1 + 48) = v8;
  *(_QWORD *)(a1 + 56) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 200LL))(v6);
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)(a1 + 64) = v9;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0x600000000LL;
  *(_QWORD *)(a1 + 152) = 0x600000000LL;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 280) = 0x10000000000LL;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_DWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 6432) = 0;
  *(_QWORD *)(a1 + 6440) = 0xFFFFFFFFLL;
  *(_DWORD *)(a1 + 6448) = 0;
  result = sub_35E6920(a1, 1);
  v11 = *(_QWORD *)(a1 + 72);
  *(_QWORD *)(a1 + 72) = result;
  if ( v11 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
  return result;
}
