// Function: sub_38C5870
// Address: 0x38c5870
//
__int64 __fastcall sub_38C5870(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 *v4; // r14
  __int64 v5; // rsi
  int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rbx

  v3 = *(__int64 **)(a1 + 8);
  v4 = &v3[4 * *(unsigned int *)(a1 + 16)];
  while ( v4 != v3 )
  {
    v5 = *v3;
    v3 += 4;
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 400LL))(a2, v5, *(v3 - 3));
    (*(void (__fastcall **)(__int64, void *, __int64))(*(_QWORD *)a2 + 400LL))(a2, &unk_452DFBC, 1);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 424LL))(a2, 0, 1);
  if ( *(_DWORD *)(a1 + 128) > 1u )
  {
    v6 = 1;
    v7 = 1;
    do
    {
      v8 = 72 * v7;
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 400LL))(
        a2,
        *(_QWORD *)(v8 + *(_QWORD *)(a1 + 120)),
        *(_QWORD *)(v8 + *(_QWORD *)(a1 + 120) + 8));
      (*(void (__fastcall **)(__int64, void *, __int64))(*(_QWORD *)a2 + 400LL))(a2, &unk_452DFBC, 1);
      sub_38DCDD0(a2, *(unsigned int *)(*(_QWORD *)(a1 + 120) + v8 + 32));
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 424LL))(a2, 0, 1);
      (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 424LL))(a2, 0, 1);
      v7 = (unsigned int)(v6 + 1);
      v6 = v7;
    }
    while ( (unsigned int)v7 < *(_DWORD *)(a1 + 128) );
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 424LL))(a2, 0, 1);
}
