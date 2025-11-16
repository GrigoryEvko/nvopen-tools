// Function: sub_3545000
// Address: 0x3545000
//
__int64 __fastcall sub_3545000(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 (*v10)(); // rcx
  __int64 (*v11)(); // rdx
  char v12; // al
  _DWORD *v13; // r12
  unsigned __int64 v14; // r14
  __int64 result; // rax

  v7 = 0;
  v8 = a2[25];
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 8) = v8;
  v9 = *a2;
  v10 = *(__int64 (**)())(*a2 + 128LL);
  if ( v10 != sub_2DAC790 )
  {
    v7 = ((__int64 (__fastcall *)(_QWORD *))v10)(a2);
    v9 = *a2;
  }
  *(_QWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 32) = a3;
  v11 = *(__int64 (**)())(v9 + 384);
  v12 = 1;
  if ( v11 != sub_3059490 )
    v12 = ((__int64 (__fastcall *)(_QWORD *))v11)(a2);
  *(_BYTE *)(a1 + 40) = v12;
  v13 = *(_DWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  *(_QWORD *)(a1 + 112) = a1 + 128;
  *(_QWORD *)(a1 + 120) = 0x100000000LL;
  *(_QWORD *)(a1 + 272) = a1 + 288;
  *(_QWORD *)(a1 + 280) = 0xC00000000LL;
  v14 = (unsigned int)v13[12];
  *(_QWORD *)(a1 + 336) = a1 + 352;
  *(_QWORD *)(a1 + 344) = 0x1000000000LL;
  if ( (unsigned int)v14 > 0x10 )
  {
    sub_C8D5F0(a1 + 336, (const void *)(a1 + 352), v14, 8u, a5, a6);
    memset(*(void **)(a1 + 336), 0, 8 * v14);
    *(_DWORD *)(a1 + 344) = v14;
    v13 = *(_DWORD **)(a1 + 8);
  }
  else
  {
    if ( v14 )
      memset((void *)(a1 + 352), 0, 8 * v14);
    *(_DWORD *)(a1 + 344) = v14;
  }
  *(_DWORD *)(a1 + 480) = 0;
  *(_DWORD *)(a1 + 484) = *v13;
  sub_3544E90(a1, (__int64)v13, a1 + 336, (__int64)v10, a5, a6);
  if ( *(int *)(a1 + 484) <= 0 )
    *(_DWORD *)(a1 + 484) = 100;
  result = LODWORD(qword_503DF40[17]);
  if ( (int)result > 0 )
    *(_DWORD *)(a1 + 484) = result;
  return result;
}
