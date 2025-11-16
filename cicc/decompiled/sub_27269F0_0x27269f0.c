// Function: sub_27269F0
// Address: 0x27269f0
//
__int64 __fastcall sub_27269F0(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        unsigned __int8 **a4,
        __int64 **a5,
        __int64 **a6)
{
  __int64 v8; // rbx
  _QWORD *v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rdx
  unsigned int v14; // ecx
  unsigned int v15; // ebx
  __int64 v17; // r15
  unsigned __int8 *v18; // rdi
  __int64 *v19; // rax
  __int64 *v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-48h]

  v8 = a3;
  v9 = (_QWORD *)sub_BD5C60(a2);
  v10 = sub_BCB2E0(v9);
  v11 = 0;
  if ( *(char *)(a2 + 7) < 0 )
    v11 = sub_BD2BC0(a2);
  v12 = v11 + 16 * v8;
  v13 = *(_QWORD *)v12;
  v14 = *(_DWORD *)(v12 + 8);
  v15 = *(_DWORD *)(v12 + 12);
  if ( *(_QWORD *)v13 != 5 )
    return 0;
  if ( *(_DWORD *)(v13 + 16) != 1734962273 )
    return 0;
  if ( *(_BYTE *)(v13 + 20) != 110 )
    return 0;
  v25 = 32LL * v14;
  v17 = v25 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) + a2;
  v18 = *(unsigned __int8 **)v17;
  *a4 = *(unsigned __int8 **)v17;
  *a4 = sub_BD3E50(v18, a2);
  v19 = sub_DD8400(*a1, *(_QWORD *)(v17 + 32));
  *a5 = v19;
  v20 = sub_DC5760(*a1, (__int64)v19, v10, 0);
  *a5 = v20;
  if ( *((_WORD *)v20 + 12) )
    return 0;
  v21 = v20[4];
  if ( *(_DWORD *)(v21 + 32) > 0x40u )
  {
    if ( (unsigned int)sub_C44630(v21 + 24) == 1 )
      goto LABEL_11;
    return 0;
  }
  v22 = *(_QWORD *)(v21 + 24);
  if ( !v22 || (v22 & (v22 - 1)) != 0 )
    return 0;
LABEL_11:
  v23 = *a1;
  if ( 32LL * v15 - v25 == 96 )
    v24 = sub_DD8400(v23, *(_QWORD *)(v17 + 64));
  else
    v24 = sub_DA2C50(v23, v10, 0, 0);
  *a6 = v24;
  *a6 = sub_DC5760(*a1, (__int64)v24, v10, 0);
  return 1;
}
