// Function: sub_E769A0
// Address: 0xe769a0
//
__int64 __fastcall sub_E769A0(__int64 a1, __int64 a2)
{
  __int64 *v3; // r14
  __int64 *v4; // r13
  __int64 v5; // rsi
  int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // r13
  int v10; // [rsp+Ch] [rbp-34h]

  v10 = 0;
  if ( *(_QWORD *)(a1 + 408) )
  {
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 512LL))(a2, *(_QWORD *)(a1 + 400));
    (*(void (__fastcall **)(__int64, void *, __int64))(*(_QWORD *)a2 + 512LL))(a2, &unk_3F801CE, 1);
    v10 = 1;
  }
  v3 = *(__int64 **)(a1 + 8);
  v4 = &v3[4 * *(unsigned int *)(a1 + 16)];
  while ( v4 != v3 )
  {
    v5 = *v3;
    v3 += 4;
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 512LL))(a2, v5, *(v3 - 3));
    (*(void (__fastcall **)(__int64, void *, __int64))(*(_QWORD *)a2 + 512LL))(a2, &unk_3F801CE, 1);
  }
  v6 = 1;
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
  v7 = 1;
  if ( *(_DWORD *)(a1 + 128) > 1u )
  {
    do
    {
      v8 = 80 * v7;
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 512LL))(
        a2,
        *(_QWORD *)(v8 + *(_QWORD *)(a1 + 120)),
        *(_QWORD *)(v8 + *(_QWORD *)(a1 + 120) + 8));
      (*(void (__fastcall **)(__int64, void *, __int64))(*(_QWORD *)a2 + 512LL))(a2, &unk_3F801CE, 1);
      sub_E98EB0(a2, (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 120) + v8 + 32) + v10), 0);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
      v7 = (unsigned int)(v6 + 1);
      v6 = v7;
    }
    while ( (unsigned int)v7 < *(_DWORD *)(a1 + 128) );
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
}
