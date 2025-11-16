// Function: sub_300BF60
// Address: 0x300bf60
//
__int64 __fastcall sub_300BF60(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // r12
  unsigned int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 (*v6)(); // rax
  unsigned __int8 v7; // r15
  __int64 v8; // rax

  v2 = -1;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 312LL)
     + 16LL
     * (*(unsigned __int16 *)(*(_QWORD *)a2 + 24LL)
      + *(_DWORD *)(*(_QWORD *)(a1 + 16) + 328LL)
      * (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 288LL) - *(_QWORD *)(*(_QWORD *)(a1 + 16) + 280LL)) >> 3));
  v3 = *(_DWORD *)(v4 + 4) >> 3;
  LODWORD(v4) = *(_DWORD *)(v4 + 8) >> 3;
  if ( (_DWORD)v4 )
  {
    _BitScanReverse64((unsigned __int64 *)&v4, (unsigned int)v4);
    v2 = 63 - (v4 ^ 0x3F);
  }
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 136LL);
  if ( v6 == sub_2DD19D0 )
    BUG();
  v7 = *(_BYTE *)(((__int64 (__fastcall *)(_QWORD))v6)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL)) + 12);
  if ( v7 < v2 )
  {
    v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v8 + 536LL))(v8, *(_QWORD *)(a1 + 24)) )
      v2 = v7;
  }
  return sub_2E77CA0(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 48LL), v3, v2);
}
