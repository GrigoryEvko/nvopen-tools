// Function: sub_E0FB60
// Address: 0xe0fb60
//
__int64 __fastcall sub_E0FB60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  unsigned __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  char v12; // dl

  v7 = *(_QWORD **)(a1 + 4096);
  v8 = v7[1] + 32LL;
  if ( v8 > 0xFEF )
  {
    v9 = (_QWORD *)malloc(4096, a2, a3, a4, a5, a6);
    if ( !v9 )
      sub_2207530(4096, a2, v10);
    *v9 = v7;
    v7 = v9;
    v9[1] = 0;
    *(_QWORD *)(a1 + 4096) = v9;
    v8 = 32;
  }
  v7[1] = v8;
  result = *(_QWORD *)(a1 + 4096) + *(_QWORD *)(*(_QWORD *)(a1 + 4096) + 8LL) - 16LL;
  *(_WORD *)(result + 8) = 16392;
  v12 = *(_BYTE *)(result + 10);
  *(_QWORD *)(result + 16) = a2;
  *(_QWORD *)(result + 24) = a3;
  *(_BYTE *)(result + 10) = v12 & 0xF0 | 5;
  *(_QWORD *)result = &unk_49DEFA8;
  return result;
}
