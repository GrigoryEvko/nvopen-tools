// Function: sub_ECA090
// Address: 0xeca090
//
__int64 __fastcall sub_ECA090(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 v5; // rax
  __int64 v6; // r12
  unsigned int v7; // r13d
  __int64 v8; // rax
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v2 = *(unsigned int *)(v1 + 128);
  if ( (_DWORD)v2
    && (v5 = *(_QWORD *)(v1 + 120) + 32 * v2 - 32, v6 = *(_QWORD *)(v5 + 16), v7 = *(_DWORD *)(v5 + 24), v6) )
  {
    v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8 + 176LL))(v8, v6, v7);
    return 0;
  }
  else
  {
    v3 = *(_QWORD *)(a1 + 8);
    v11 = 1;
    v9 = ".previous without corresponding .section";
    v10 = 3;
    return sub_ECE0E0(v3, &v9, 0, 0);
  }
}
