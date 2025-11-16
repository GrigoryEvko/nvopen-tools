// Function: sub_E0FEB0
// Address: 0xe0feb0
//
__int64 __fastcall sub_E0FEB0(__int64 a1, const char *a2, __int64 a3)
{
  size_t v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r14
  size_t v10; // r15
  unsigned __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 result; // rax
  char v15; // dl

  v4 = strlen(a2);
  v9 = *(_QWORD **)(a1 + 4096);
  v10 = v4;
  v11 = v9[1] + 48LL;
  if ( v11 > 0xFEF )
  {
    v12 = (_QWORD *)malloc(4096, a2, v5, v6, v7, v8);
    if ( !v12 )
      sub_2207530(4096, a2, v13);
    *v12 = v9;
    v9 = v12;
    v12[1] = 0;
    *(_QWORD *)(a1 + 4096) = v12;
    v11 = 48;
  }
  v9[1] = v11;
  result = *(_QWORD *)(a1 + 4096) + *(_QWORD *)(*(_QWORD *)(a1 + 4096) + 8LL) - 32LL;
  *(_WORD *)(result + 8) = 16405;
  v15 = *(_BYTE *)(result + 10);
  *(_QWORD *)(result + 16) = v10;
  *(_QWORD *)(result + 24) = a2;
  *(_QWORD *)(result + 32) = a3;
  *(_BYTE *)(result + 10) = v15 & 0xF0 | 5;
  *(_QWORD *)result = &unk_49DF608;
  return result;
}
