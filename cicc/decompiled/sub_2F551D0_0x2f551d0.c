// Function: sub_2F551D0
// Address: 0x2f551d0
//
void __fastcall sub_2F551D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // ecx
  unsigned int v3; // ecx
  unsigned int v4; // ecx
  unsigned int v5; // ecx
  unsigned int v6; // ecx
  unsigned int v7; // ecx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v20[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v21; // [rsp+20h] [rbp-60h]
  _QWORD v22[10]; // [rsp+30h] [rbp-50h] BYREF

  v2 = *(_DWORD *)(a1 + 12);
  if ( !v2 )
    goto LABEL_2;
  sub_B169E0(v19, "NumSpills", 9, v2);
  v8 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v8, " spills ", 8u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  sub_B16720((__int64)v19, "TotalSpillsCost", 15, *(float *)(a1 + 32));
  v9 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v9, " total spills cost ", 0x13u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] == v20 )
  {
LABEL_2:
    v3 = *(_DWORD *)(a1 + 16);
    if ( !v3 )
      goto LABEL_3;
  }
  else
  {
    j_j___libc_free_0(v19[0]);
    v3 = *(_DWORD *)(a1 + 16);
    if ( !v3 )
      goto LABEL_3;
  }
  sub_B169E0(v19, "NumFoldedSpills", 15, v3);
  v10 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v10, " folded spills ", 0xFu);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  sub_B16720((__int64)v19, "TotalFoldedSpillsCost", 21, *(float *)(a1 + 36));
  v11 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v11, " total folded spills cost ", 0x1Au);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
  {
    j_j___libc_free_0(v19[0]);
    v4 = *(_DWORD *)a1;
    if ( !*(_DWORD *)a1 )
      goto LABEL_4;
    goto LABEL_24;
  }
LABEL_3:
  v4 = *(_DWORD *)a1;
  if ( !*(_DWORD *)a1 )
    goto LABEL_4;
LABEL_24:
  sub_B169E0(v19, "NumReloads", 10, v4);
  v12 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v12, " reloads ", 9u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  sub_B16720((__int64)v19, "TotalReloadsCost", 16, *(float *)(a1 + 24));
  v13 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v13, " total reloads cost ", 0x14u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
  {
    j_j___libc_free_0(v19[0]);
    v5 = *(_DWORD *)(a1 + 4);
    if ( !v5 )
      goto LABEL_5;
    goto LABEL_32;
  }
LABEL_4:
  v5 = *(_DWORD *)(a1 + 4);
  if ( !v5 )
    goto LABEL_5;
LABEL_32:
  sub_B169E0(v19, "NumFoldedReloads", 16, v5);
  v14 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v14, " folded reloads ", 0x10u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  sub_B16720((__int64)v19, "TotalFoldedReloadsCost", 22, *(float *)(a1 + 28));
  v15 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v15, " total folded reloads cost ", 0x1Bu);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
  {
    j_j___libc_free_0(v19[0]);
    v6 = *(_DWORD *)(a1 + 8);
    if ( !v6 )
      goto LABEL_6;
    goto LABEL_40;
  }
LABEL_5:
  v6 = *(_DWORD *)(a1 + 8);
  if ( !v6 )
    goto LABEL_6;
LABEL_40:
  sub_B169E0(v19, "NumZeroCostFoldedReloads", 24, v6);
  v16 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v16, " zero cost folded reloads ", 0x1Au);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
  {
    j_j___libc_free_0(v19[0]);
    v7 = *(_DWORD *)(a1 + 20);
    if ( !v7 )
      return;
    goto LABEL_44;
  }
LABEL_6:
  v7 = *(_DWORD *)(a1 + 20);
  if ( !v7 )
    return;
LABEL_44:
  sub_B169E0(v19, "NumVRCopies", 11, v7);
  v17 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v17, " virtual registers copies ", 0x1Au);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
  sub_B16720((__int64)v19, "TotalCopiesCost", 15, *(float *)(a1 + 40));
  v18 = sub_2F55120(a2, (__int64)v19);
  sub_B18290(v18, " total copies cost ", 0x13u);
  if ( v21 != v22 )
    j_j___libc_free_0((unsigned __int64)v21);
  if ( (_QWORD *)v19[0] != v20 )
    j_j___libc_free_0(v19[0]);
}
