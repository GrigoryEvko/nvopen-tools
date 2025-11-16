// Function: sub_E18460
// Address: 0xe18460
//
__int64 __fastcall sub_E18460(__int64 *a1, __int64 *a2)
{
  _BYTE *v2; // rax
  _BYTE *v3; // rdx
  char v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  void *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  char v17; // dl
  __int64 v18; // [rsp-40h] [rbp-40h] BYREF

  v2 = (_BYTE *)*a1;
  v3 = (_BYTE *)a1[1];
  if ( v3 == (_BYTE *)*a1 )
    return 0;
  while ( 1 )
  {
    if ( *v2 != 87 )
      return 0;
    v5 = 0;
    *a1 = (__int64)(v2 + 1);
    if ( v2 + 1 != v3 && v2[1] == 80 )
    {
      v5 = 1;
      *a1 = (__int64)(v2 + 2);
    }
    v10 = sub_E12DE0(a1);
    if ( !v10 )
      break;
    v11 = *a2;
    v12 = sub_E0E790((__int64)(a1 + 102), 40, v6, v7, v8, v9);
    if ( v12 )
    {
      *(_QWORD *)(v12 + 16) = v11;
      *(_WORD *)(v12 + 8) = 16411;
      v17 = *(_BYTE *)(v12 + 10);
      *(_QWORD *)(v12 + 24) = v10;
      *(_BYTE *)(v12 + 32) = v5;
      *(_BYTE *)(v12 + 10) = v17 & 0xF0 | 5;
      v13 = &unk_49DF788;
      *(_QWORD *)v12 = &unk_49DF788;
    }
    *a2 = v12;
    v18 = v12;
    sub_E18380((__int64)(a1 + 37), &v18, (__int64)v13, v14, v15, v16);
    v2 = (_BYTE *)*a1;
    v3 = (_BYTE *)a1[1];
    if ( (_BYTE *)*a1 == v3 )
      return 0;
  }
  return 1;
}
