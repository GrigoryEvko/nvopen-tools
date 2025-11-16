// Function: sub_1B1B030
// Address: 0x1b1b030
//
__int64 __fastcall sub_1B1B030(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rcx
  _BOOL4 v7; // eax
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 *v12; // rax
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // r8
  _BOOL4 v16; // eax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-90h]
  __int64 v25; // [rsp+8h] [rbp-88h]
  _QWORD v26[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v27; // [rsp+20h] [rbp-70h]
  char *v28; // [rsp+40h] [rbp-50h]
  char v29; // [rsp+50h] [rbp-40h] BYREF

  if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(a1 + 40) || (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) != 2 )
    return 0;
  v6 = a1 - 48;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v6 = *(_QWORD *)(a1 - 8);
  v7 = sub_1377F70(a2 + 56, *(_QWORD *)(v6 + 24LL * *(unsigned int *)(a1 + 56) + 8));
  v8 = a3;
  if ( v7 )
  {
    v9 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
       ? *(__int64 **)(a1 - 8)
       : (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v10 = *v9;
    v11 = v9[3];
  }
  else
  {
    v12 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
        ? *(__int64 **)(a1 - 8)
        : (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v10 = v12[3];
    v11 = *v12;
  }
  v13 = *(_BYTE *)(v10 + 16);
  if ( (unsigned __int8)(v13 - 35) > 0x11u )
    return 0;
  if ( v13 == 36 )
  {
    v15 = *(_QWORD *)(v10 - 48);
    v22 = *(_QWORD *)(v10 - 24);
    if ( v15 && a1 == v15 )
    {
      if ( v22 )
      {
        v15 = *(_QWORD *)(v10 - 24);
        goto LABEL_20;
      }
      return 0;
    }
    if ( v22 == 0 || a1 != v22 )
      return 0;
  }
  else
  {
    if ( v13 != 38 )
      return 0;
    v14 = *(_QWORD *)(v10 - 48);
    if ( a1 != v14 || !v14 )
      return 0;
    v15 = *(_QWORD *)(v10 - 24);
  }
  if ( !v15 )
    return 0;
LABEL_20:
  if ( *(_BYTE *)(v15 + 16) > 0x17u )
  {
    v23 = a3;
    v25 = v15;
    v16 = sub_1377F70(a2 + 56, *(_QWORD *)(v15 + 40));
    v15 = v25;
    v8 = v23;
    if ( v16 )
      return 0;
  }
  v17 = sub_145DC80(v8, v15);
  sub_1B16880((__int64)v26, v11, 3, v17, v10, 0);
  sub_1B15B50(a4, (__int64)v26, v18, v19, v20, v21);
  if ( v28 != &v29 )
    _libc_free((unsigned __int64)v28);
  if ( v27 != 0 && v27 != -8 && v27 != -16 )
    sub_1649B30(v26);
  return 1;
}
