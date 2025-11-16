// Function: sub_3906230
// Address: 0x3906230
//
__int64 __fastcall sub_3906230(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r8
  __int64 v3; // r9
  __int64 v4; // rax
  int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r8
  __int64 v10; // rdi
  const char *v11; // [rsp+0h] [rbp-30h] BYREF
  char v12; // [rsp+10h] [rbp-20h]
  char v13; // [rsp+11h] [rbp-1Fh]

  v1 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  v4 = *(unsigned int *)(v1 + 120);
  if ( (unsigned int)v4 <= 1 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v13 = 1;
    v11 = ".popsection without corresponding .pushsection";
    v12 = 3;
    return sub_3909CF0(v10, &v11, 0, 0, v2, v3);
  }
  else
  {
    v5 = *(_DWORD *)(v1 + 120);
    v6 = *(_QWORD *)(v1 + 112) + 32 * v4;
    v7 = *(_QWORD *)(v6 - 64);
    v8 = *(_QWORD *)(v6 - 56);
    if ( *(_QWORD *)(v6 - 32) != v7 || *(_QWORD *)(v6 - 24) != v8 )
    {
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v1 + 152LL))(v1, v7, v8);
      v5 = *(_DWORD *)(v1 + 120);
    }
    *(_DWORD *)(v1 + 120) = v5 - 1;
    return 0;
  }
}
