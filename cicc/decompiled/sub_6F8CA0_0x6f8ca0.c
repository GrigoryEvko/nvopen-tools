// Function: sub_6F8CA0
// Address: 0x6f8ca0
//
__int64 __fastcall sub_6F8CA0(__int64 a1, __int64 a2, __m128i *a3, _QWORD *a4, _DWORD *a5, _QWORD *a6)
{
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 *v14; // [rsp+0h] [rbp-50h]
  __int64 v15; // [rsp+0h] [rbp-50h]

  v7 = *(_QWORD *)a1;
  v8 = sub_6E3DA0(*(_QWORD *)a1, 0);
  v14 = *(__int64 **)(v7 + 72);
  result = sub_6F85E0(v14, a1, 2048, (_QWORD *)a2, a3, v9);
  if ( !*(_BYTE *)(a1 + 56) )
  {
    v13 = v14[2];
    if ( (*(_BYTE *)(a2 + 18) & 1) == 0 && *(_BYTE *)(v7 + 56) != 105 )
    {
      v15 = v14[2];
      sub_6DB9F0((__int64 *)a1, (_QWORD *)a2, a3, v13, v11, v12);
      v13 = *(_QWORD *)(v15 + 16);
    }
    *(_QWORD *)(a1 + 16) = v13;
    *a4 = *(_QWORD *)(v8 + 356);
    *a5 = *(_DWORD *)(v8 + 364);
    if ( a6 )
      *a6 = *(_QWORD *)(v8 + 368);
    result = qword_4D03C50;
    if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x20) != 0 && v7 == unk_4D03C40 )
      *(_BYTE *)(qword_4D03C50 + 20LL) |= 8u;
  }
  return result;
}
