// Function: sub_3902B10
// Address: 0x3902b10
//
__int64 __fastcall sub_3902B10(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  const char *v11; // [rsp+0h] [rbp-40h] BYREF
  char v12; // [rsp+10h] [rbp-30h]
  char v13; // [rsp+11h] [rbp-2Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v4 = *(unsigned int *)(v1 + 120);
  if ( (_DWORD)v4
    && (v7 = *(_QWORD *)(v1 + 112) + 32 * v4 - 32, v8 = *(_QWORD *)(v7 + 16), v9 = *(_QWORD *)(v7 + 24), v8) )
  {
    v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v10 + 160LL))(v10, v8, v9);
    return 0;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 8);
    v13 = 1;
    v11 = ".previous without corresponding .section";
    v12 = 3;
    return sub_3909CF0(v5, &v11, 0, 0, v2, v3);
  }
}
