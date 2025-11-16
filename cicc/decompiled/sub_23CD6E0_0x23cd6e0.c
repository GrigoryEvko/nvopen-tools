// Function: sub_23CD6E0
// Address: 0x23cd6e0
//
__int64 __fastcall sub_23CD6E0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v3 = (__int64 *)(a1 + 96);
  *((_OWORD *)v3 - 4) = 0;
  *(v3 - 11) = 0;
  *(v3 - 10) = 0;
  *(v3 - 12) = (__int64)&unk_4A162C8;
  *(v3 - 9) = 0;
  *(v3 - 2) = 0;
  *(v3 - 1) = 0;
  *v3 = 0;
  v3[1] = 0;
  v3[2] = 0;
  v3[3] = 0;
  v3[4] = 0;
  v3[5] = 0;
  v3[6] = 0;
  v3[7] = 0;
  v3[8] = 0;
  v3[9] = 0;
  *((_OWORD *)v3 - 3) = 0;
  *((_OWORD *)v3 - 2) = 0;
  v6 = a2;
  sub_23CD5A0(v3, 0);
  *(_QWORD *)(a1 + 208) = 0;
  *(_OWORD *)(a1 + 176) = 0;
  *(_OWORD *)(a1 + 192) = 0;
  sub_2210B10((_OWORD *)(a1 + 216));
  sub_2210B10((_OWORD *)(a1 + 264));
  v4 = v6;
  *(_BYTE *)(a1 + 352) = 1;
  *(_DWORD *)(a1 + 312) = 0;
  *(_QWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 356) = v4;
  result = sub_C95AE0((__int64)&v6);
  *(_DWORD *)(a1 + 364) = result;
  return result;
}
