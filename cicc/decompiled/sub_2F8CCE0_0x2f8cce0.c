// Function: sub_2F8CCE0
// Address: 0x2f8cce0
//
void __fastcall sub_2F8CCE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rcx
  _QWORD *v6; // rax
  __int64 v7; // r9
  _QWORD *v8; // r12
  char v9; // si
  __int64 v10; // rdx
  _QWORD *v11; // r14
  __int64 v12; // r10
  char **v13; // r8
  int v14; // eax
  int v15; // esi
  __int64 v16; // rbx
  __int64 v17; // r9
  char **v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // r8
  char v21; // al
  int v22; // eax
  int v23; // eax
  char v24; // al
  int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-50h]
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-38h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+18h] [rbp-38h]

  v3 = 0x1745D1745D1745DLL;
  *a1 = a3;
  a1[1] = 0;
  if ( a3 <= 0x1745D1745D1745DLL )
    v3 = a3;
  a1[2] = 0;
  if ( a3 <= 0 )
    return;
  v4 = (__int64)a1;
  while ( 1 )
  {
    v30 = v4;
    v6 = (_QWORD *)sub_2207800(88 * v3);
    v4 = v30;
    v8 = v6;
    if ( v6 )
      break;
    v3 >>= 1;
    if ( !v3 )
      return;
  }
  v9 = *(_BYTE *)(a2 + 12);
  v10 = *(_QWORD *)a2;
  v11 = &v6[11 * v3];
  v12 = a2 + 16;
  v13 = (char **)(v6 + 2);
  *v6 = *(_QWORD *)a2;
  v14 = *(_DWORD *)(a2 + 8);
  *((_BYTE *)v8 + 12) = v9;
  v8[2] = v8 + 4;
  v15 = *(_DWORD *)(a2 + 24);
  *((_DWORD *)v8 + 2) = v14;
  v8[3] = 0x600000000LL;
  if ( v15 )
  {
    sub_2F8ABB0((__int64)(v8 + 2), (char **)(a2 + 16), v10, v30, (__int64)v13, v7);
    v10 = *v8;
    v14 = *((_DWORD *)v8 + 2);
    v4 = v30;
    v12 = a2 + 16;
    v13 = (char **)(v8 + 2);
  }
  v16 = (__int64)(v8 + 11);
  *((_DWORD *)v8 + 20) = *(_DWORD *)(a2 + 80);
  if ( v11 == v8 + 11 )
  {
    v17 = (__int64)v8;
    goto LABEL_15;
  }
  while ( 1 )
  {
    *(_DWORD *)(v16 + 8) = v14;
    v21 = *(_BYTE *)(v16 - 76);
    v20 = v16 - 88;
    v17 = v16;
    *(_QWORD *)v16 = v10;
    *(_BYTE *)(v16 + 12) = v21;
    *(_QWORD *)(v16 + 16) = v16 + 32;
    v22 = *(_DWORD *)(v16 - 64);
    *(_DWORD *)(v16 + 24) = 0;
    *(_DWORD *)(v16 + 28) = 6;
    if ( !v22 )
      break;
    v18 = (char **)(v16 - 72);
    v19 = v16 + 16;
    v26 = v4;
    v27 = v16 - 88;
    v28 = v16;
    v16 += 88;
    v31 = v12;
    sub_2F8ABB0(v19, v18, v10, v4, v20, v17);
    v4 = v26;
    v20 = v27;
    v17 = v28;
    *(_DWORD *)(v16 - 8) = *(_DWORD *)(v16 - 96);
    v12 = v31;
    if ( v11 == (_QWORD *)v16 )
      goto LABEL_14;
LABEL_11:
    v10 = *(_QWORD *)(v16 - 88);
    v14 = *(_DWORD *)(v16 - 80);
  }
  v23 = *(_DWORD *)(v16 - 8);
  v16 += 88;
  *(_DWORD *)(v16 - 8) = v23;
  if ( v11 != (_QWORD *)v16 )
    goto LABEL_11;
LABEL_14:
  v10 = *(_QWORD *)(v20 + 88);
  v14 = *(_DWORD *)(v20 + 96);
  v13 = (char **)(v20 + 104);
LABEL_15:
  *(_DWORD *)(a2 + 8) = v14;
  v24 = *(_BYTE *)(v17 + 12);
  *(_QWORD *)a2 = v10;
  *(_BYTE *)(a2 + 12) = v24;
  v29 = v4;
  v32 = v17;
  sub_2F8ABB0(v12, v13, v10, v4, (__int64)v13, v17);
  v25 = *(_DWORD *)(v32 + 80);
  *(_QWORD *)(v29 + 16) = v8;
  *(_QWORD *)(v29 + 8) = v3;
  *(_DWORD *)(a2 + 80) = v25;
}
