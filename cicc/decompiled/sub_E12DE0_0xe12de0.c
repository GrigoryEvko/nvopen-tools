// Function: sub_E12DE0
// Address: 0xe12de0
//
__int64 __fastcall sub_E12DE0(__int64 *a1)
{
  __int64 v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // r13
  unsigned __int64 v6; // r12
  _QWORD *v7; // r14
  unsigned __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 result; // rax
  char v12; // dl
  __int64 v13[5]; // [rsp+8h] [rbp-28h] BYREF

  v13[0] = 0;
  if ( (unsigned __int8)sub_E0EF70(a1, v13) )
    return 0;
  v5 = *a1;
  v6 = v13[0];
  if ( (unsigned __int64)(a1[1] - *a1) < v13[0] || !v13[0] )
    return 0;
  *a1 = v5 + v13[0];
  if ( v6 > 9 && *(_QWORD *)v5 == 0x5F4C41424F4C475FLL && *(_WORD *)(v5 + 8) == 20063 )
    return sub_E0FD70((__int64)(a1 + 102), "(anonymous namespace)");
  v7 = (_QWORD *)a1[614];
  v8 = v7[1] + 32LL;
  if ( v8 > 0xFEF )
  {
    v9 = (_QWORD *)malloc(4096, v13, v1, v2, v3, v4);
    if ( !v9 )
      sub_2207530(4096, v13, v10);
    *v9 = v7;
    v7 = v9;
    v9[1] = 0;
    a1[614] = (__int64)v9;
    v8 = 32;
  }
  v7[1] = v8;
  result = a1[614] + *(_QWORD *)(a1[614] + 8) - 16;
  *(_WORD *)(result + 8) = 16392;
  v12 = *(_BYTE *)(result + 10);
  *(_QWORD *)(result + 16) = v6;
  *(_QWORD *)(result + 24) = v5;
  *(_BYTE *)(result + 10) = v12 & 0xF0 | 5;
  *(_QWORD *)result = &unk_49DEFA8;
  return result;
}
