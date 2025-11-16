// Function: sub_E0FD70
// Address: 0xe0fd70
//
__int64 __fastcall sub_E0FD70(__int64 a1, const char *a2)
{
  size_t v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  _QWORD *v7; // r13
  size_t v8; // r14
  unsigned __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 result; // rax
  char v13; // dl

  v2 = strlen(a2);
  v7 = *(_QWORD **)(a1 + 4096);
  v8 = v2;
  v9 = v7[1] + 32LL;
  if ( v9 > 0xFEF )
  {
    v10 = (_QWORD *)malloc(4096, a2, v3, v4, v5, v6);
    if ( !v10 )
      sub_2207530(4096, a2, v11);
    *v10 = v7;
    v7 = v10;
    v10[1] = 0;
    *(_QWORD *)(a1 + 4096) = v10;
    v9 = 32;
  }
  v7[1] = v9;
  result = *(_QWORD *)(a1 + 4096) + *(_QWORD *)(*(_QWORD *)(a1 + 4096) + 8LL) - 16LL;
  *(_WORD *)(result + 8) = 16392;
  v13 = *(_BYTE *)(result + 10);
  *(_QWORD *)(result + 16) = v8;
  *(_QWORD *)(result + 24) = a2;
  *(_BYTE *)(result + 10) = v13 & 0xF0 | 5;
  *(_QWORD *)result = &unk_49DEFA8;
  return result;
}
