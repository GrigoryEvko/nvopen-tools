// Function: sub_2FC8C00
// Address: 0x2fc8c00
//
void __fastcall sub_2FC8C00(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 i; // r13
  void (__fastcall *v5)(__int64, __int64, __int64); // rax
  unsigned __int16 *v6; // r12
  unsigned __int16 *v7; // r15
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // r15
  __int64 v11; // rsi

  v2 = *(_QWORD *)(a1 + 16);
  for ( i = *(_QWORD *)(a1 + 8); v2 != i; i += 192 )
  {
    while ( 1 )
    {
      v5 = *(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 536LL);
      if ( *(_DWORD *)(i + 24) <= 0xFFFFu && *(_DWORD *)(i + 136) <= 0xFFFFu )
        break;
      i += 192;
      v5(a2, -1, 8);
      sub_E9A5B0(a2, *(unsigned __int8 **)(i - 192));
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 4);
      if ( v2 == i )
        return;
    }
    v5(a2, *(_QWORD *)(i + 8), 8);
    sub_E9A5B0(a2, *(unsigned __int8 **)i);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(unsigned int *)(i + 24), 2);
    v6 = *(unsigned __int16 **)(i + 16);
    v7 = &v6[6 * *(unsigned int *)(i + 24)];
    while ( v7 != v6 )
    {
      v8 = *v6;
      v6 += 6;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 536LL))(a2, v8, 1);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(v6 - 5), 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(v6 - 4), 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *((int *)v6 - 1), 4);
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a2 + 608LL))(a2, 3, 0, 1, 0);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(unsigned int *)(i + 136), 2);
    v9 = *(_QWORD *)(i + 128);
    v10 = v9 + 6LL * *(unsigned int *)(i + 136);
    while ( v9 != v10 )
    {
      v11 = *(unsigned __int16 *)(v9 + 2);
      v9 += 6;
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 536LL))(a2, v11, 2);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(unsigned __int16 *)(v9 - 2), 1);
    }
    (*(void (__fastcall **)(__int64, __int64, _QWORD, __int64, _QWORD))(*(_QWORD *)a2 + 608LL))(a2, 3, 0, 1, 0);
  }
}
