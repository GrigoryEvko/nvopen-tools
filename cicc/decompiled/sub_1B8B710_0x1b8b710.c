// Function: sub_1B8B710
// Address: 0x1b8b710
//
char __fastcall sub_1B8B710(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // r8
  char v11; // r9
  char v12; // r10
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rcx
  int v20; // ebx
  __int64 v21; // rbx
  __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  _QWORD *v26; // rdx
  unsigned __int64 v27; // rax
  unsigned int v28; // [rsp+4h] [rbp-4Ch]
  __int64 v29; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v29 = a2;
  v30[0] = &v29;
  v6 = sub_1B7F330(a1, a3);
  v7 = sub_1B7F330(a1, a4);
  if ( !(v7 | v6) )
  {
LABEL_14:
    v20 = sub_1B8B3F0(v30, a3);
    return v20 < (int)sub_1B8B3F0(v30, a4);
  }
  if ( !v6 )
    return 1;
  if ( !v7 )
    return 0;
  v8 = *(_DWORD *)(v6 + 20) & 0xFFFFFFF;
  v9 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
  if ( (_DWORD)v8 != (_DWORD)v9 )
    return (unsigned int)v8 < (unsigned int)v9;
  v11 = *(_BYTE *)(v7 + 23) & 0x40;
  v12 = *(_BYTE *)(v6 + 23) & 0x40;
  v28 = v8 - 1;
  if ( (_DWORD)v8 != 1 )
  {
    v13 = 24LL * (unsigned int)v8;
    v14 = v6 - v13;
    v15 = v7 - v13;
    v16 = 0;
    do
    {
      v17 = v14;
      if ( v12 )
        v17 = *(_QWORD *)(v6 - 8);
      v18 = *(_QWORD *)(v17 + v16);
      v19 = v15;
      if ( v11 )
        v19 = *(_QWORD *)(v7 - 8);
      if ( *(_QWORD *)(v19 + v16) != v18 )
        goto LABEL_14;
      v16 += 24;
    }
    while ( 8 * (3LL * (unsigned int)(v8 - 2) + 3) != v16 );
  }
  if ( v12 )
    v21 = *(_QWORD *)(v6 - 8);
  else
    v21 = v6 - 24 * v8;
  v22 = 0;
  v23 = *(_QWORD *)(v21 + 24LL * v28);
  if ( *(_BYTE *)(v23 + 16) == 13 )
    v22 = *(_QWORD *)(v21 + 24LL * v28);
  if ( v11 )
    v24 = *(_QWORD *)(v7 - 8);
  else
    v24 = v7 - 24 * v9;
  v25 = *(_QWORD *)(v24 + 24LL * v28);
  if ( *(_BYTE *)(v25 + 16) != 13 || !v22 )
    return v25 > v23;
  v26 = *(_QWORD **)(v22 + 24);
  if ( *(_DWORD *)(v22 + 32) > 0x40u )
    v26 = (_QWORD *)*v26;
  if ( *(_DWORD *)(v25 + 32) <= 0x40u )
    v27 = *(_QWORD *)(v25 + 24);
  else
    v27 = **(_QWORD **)(v25 + 24);
  return v27 > (unsigned __int64)v26;
}
