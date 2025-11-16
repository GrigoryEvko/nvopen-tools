// Function: sub_1687180
// Address: 0x1687180
//
_QWORD *__fastcall sub_1687180(__int64 a1, __int64 a2)
{
  char v2; // r13
  unsigned int v3; // ebx
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rdx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  _QWORD *v10; // r12
  char *v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // edx
  int v16; // ecx
  _QWORD *v17; // rdi
  int v18; // r8d
  int v19; // r9d
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rdx
  int v23; // ecx
  _QWORD *v24; // rbx
  int v25; // r8d
  int v26; // r9d
  __int64 v27; // rdi
  int v28; // edx
  int v29; // ecx
  _QWORD *v30; // rbx
  int v31; // r8d
  int v32; // r9d
  char v34; // [rsp+0h] [rbp-30h]

  if ( !(_DWORD)a1 )
    a1 = 1;
  v2 = sub_1683C90(a1);
  v3 = 1 << v2;
  v5 = *(_QWORD *)(sub_1689050(a1, a2, v4) + 24);
  v10 = sub_1685080(v5, 112);
  if ( !v10 )
    sub_1683C30(v5, 112, v6, v7, v8, v9, v34);
  *v10 = 0;
  v11 = (char *)((unsigned __int64)(v10 + 1) & 0xFFFFFFFFFFFFFFF8LL);
  v10[13] = 0;
  v12 = (unsigned int)((_DWORD)v10 - (_DWORD)v11 + 112) >> 3;
  memset(v11, 0, 8 * v12);
  v13 = sub_1689050(&v11[8 * v12], 112, v6);
  v14 = 8LL * v3;
  v17 = sub_1685080(*(_QWORD *)(v13 + 24), v14);
  if ( !v17 )
  {
    sub_1683C30(0, v14, v15, v16, v18, v19, v34);
    v17 = 0;
  }
  v10[13] = v17;
  memset(v17, 0, 8LL * v3);
  *((_DWORD *)v10 + 10) = v3 - 1;
  v10[8] = (unsigned int)(4 << v2);
  v21 = *(_QWORD *)(sub_1689050(v17, 0, v20) + 24);
  v24 = sub_1685080(v21, 8);
  if ( !v24 )
    sub_1683C30(v21, 8, v22, v23, v25, v26, v34);
  v10[11] = v24;
  *v24 = 0;
  *((_BYTE *)v10 + 84) &= 0xFCu;
  *((_DWORD *)v10 + 19) = 1;
  v27 = *(_QWORD *)(sub_1689050(v21, 8, v22) + 24);
  v30 = sub_1685080(v27, 4);
  if ( !v30 )
    sub_1683C30(v27, 4, v28, v29, v31, v32, v34);
  v10[12] = v30;
  *(_DWORD *)v30 = 0;
  *((_DWORD *)v10 + 20) = 1;
  *((_BYTE *)v10 + 84) &= 0xF3u;
  return v10;
}
