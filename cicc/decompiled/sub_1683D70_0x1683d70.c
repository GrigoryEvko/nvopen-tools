// Function: sub_1683D70
// Address: 0x1683d70
//
__int64 __fastcall sub_1683D70(unsigned int a1)
{
  char v1; // r13
  unsigned int v2; // ebx
  __int64 v3; // rdi
  int v4; // edx
  int v5; // ecx
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // edx
  int v12; // ecx
  void *v13; // rdi
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rdi
  int v17; // edx
  int v18; // ecx
  _OWORD *v19; // rbx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rdi
  int v23; // edx
  int v24; // ecx
  _DWORD *v25; // rbx
  int v26; // r8d
  int v27; // r9d
  char v29; // [rsp+0h] [rbp-30h]

  if ( !a1 )
    a1 = 1;
  v1 = sub_1683C90(a1);
  v2 = 1 << v1;
  v3 = *(_QWORD *)(sub_1689050() + 24);
  v8 = sub_1685080(v3, 112);
  if ( !v8 )
    sub_1683C30(v3, 112, v4, v5, v6, v7, v29);
  *(_QWORD *)v8 = 0;
  *(_QWORD *)(v8 + 104) = 0;
  memset(
    (void *)((v8 + 8) & 0xFFFFFFFFFFFFFFF8LL),
    0,
    8LL * (((unsigned int)v8 - (((_DWORD)v8 + 8) & 0xFFFFFFF8) + 112) >> 3));
  v9 = sub_1689050();
  v10 = 8LL * v2;
  v13 = (void *)sub_1685080(*(_QWORD *)(v9 + 24), v10);
  if ( !v13 )
  {
    sub_1683C30(0, v10, v11, v12, v14, v15, v29);
    v13 = 0;
  }
  *(_QWORD *)(v8 + 104) = v13;
  memset(v13, 0, 8LL * v2);
  *(_DWORD *)(v8 + 40) = v2 - 1;
  *(_QWORD *)(v8 + 64) = (unsigned int)(4 << v1);
  v16 = *(_QWORD *)(sub_1689050() + 24);
  v19 = (_OWORD *)sub_1685080(v16, 16);
  if ( !v19 )
    sub_1683C30(v16, 16, v17, v18, v20, v21, v29);
  *(_QWORD *)(v8 + 88) = v19;
  *v19 = 0;
  *(_BYTE *)(v8 + 84) &= 0xFCu;
  *(_DWORD *)(v8 + 76) = 1;
  v22 = *(_QWORD *)(sub_1689050() + 24);
  v25 = (_DWORD *)sub_1685080(v22, 4);
  if ( !v25 )
    sub_1683C30(v22, 4, v23, v24, v26, v27, v29);
  *(_QWORD *)(v8 + 96) = v25;
  *v25 = 0;
  *(_DWORD *)(v8 + 80) = 1;
  *(_BYTE *)(v8 + 84) &= 0xF3u;
  return v8;
}
