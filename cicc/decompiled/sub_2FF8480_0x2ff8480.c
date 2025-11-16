// Function: sub_2FF8480
// Address: 0x2ff8480
//
__int64 __fastcall sub_2FF8480(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  unsigned int v7; // r15d
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (*v11)(); // rax
  _WORD *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rsi
  unsigned __int16 *v15; // rax
  unsigned __int16 *v16; // rcx

  if ( *(_BYTE *)(a1 + 72) )
    return (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 1144LL))(*(_QWORD *)(a1 + 200));
  if ( *(_DWORD *)(a1 + 4) <= 1u )
    return 1;
  v7 = *(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * a3 + 8);
  v8 = sub_2E88D60(a2);
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v8 + 16) + 200LL))(*(_QWORD *)(v8 + 16));
  if ( (unsigned int)sub_2E89C70(a4, v7, v9, 0) != -1
    || (v10 = *(_QWORD *)(a1 + 200), v11 = *(__int64 (**)())(*(_QWORD *)v10 + 920LL), v11 == sub_2DB1B30)
    || !((unsigned __int8 (__fastcall *)(__int64, __int64))v11)(v10, a4) )
  {
    if ( !sub_2FF7B70(a1) )
      return 0;
    v12 = sub_2FF7DB0(a1, a2);
    if ( (*v12 & 0x1FFF) == 0x1FFF )
      return 0;
    v13 = (unsigned __int16)v12[1];
    v14 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 176LL);
    v15 = (unsigned __int16 *)(v14 + 6 * v13);
    v16 = (unsigned __int16 *)(v14 + 6 * (v13 + (unsigned __int16)v12[2]));
    if ( v16 == v15 )
      return 0;
    while ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 32LL * *v15 + 16) )
    {
      v15 += 3;
      if ( v15 == v16 )
        return 0;
    }
    return 1;
  }
  return sub_2FF8080(a1, a2, 1);
}
