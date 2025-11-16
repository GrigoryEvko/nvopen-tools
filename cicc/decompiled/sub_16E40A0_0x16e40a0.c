// Function: sub_16E40A0
// Address: 0x16e40a0
//
__int64 __fastcall sub_16E40A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 result; // rax

  v7 = a4;
  sub_16E3E10((_QWORD *)a1, a4);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  v11 = 16;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49EF808;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v12 = sub_22077B0(16);
  v16 = v12;
  if ( v12 )
  {
    v7 = a2;
    v11 = v12;
    sub_16FE750(v12, a2, a3, a1 + 16, 0, a1 + 96);
  }
  *(_QWORD *)(a1 + 80) = v16;
  *(_QWORD *)(a1 + 88) = 0;
  *(_DWORD *)(a1 + 96) = 0;
  v17 = sub_2241E40(v11, v7, v13, v14, v15);
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 104) = v17;
  *(_QWORD *)(a1 + 128) = a1 + 144;
  *(_QWORD *)(a1 + 136) = 0x400000000LL;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 1;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  if ( a5 )
  {
    *(_QWORD *)(a1 + 64) = a5;
    *(_QWORD *)(a1 + 72) = a6;
  }
  result = sub_16FF550(v16);
  *(_QWORD *)(a1 + 216) = result;
  return result;
}
