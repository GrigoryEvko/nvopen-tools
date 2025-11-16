// Function: sub_1A69C30
// Address: 0x1a69c30
//
__int64 __fastcall sub_1A69C30(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  char v6; // al
  __int16 v7; // ax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // r15
  unsigned int v16; // eax
  __int64 v17; // r9
  __int64 *v18; // rax
  __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // [rsp+8h] [rbp-58h]
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-48h]
  unsigned __int64 v28; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-38h]

  v6 = *(_BYTE *)(a3 + 16);
  if ( v6 == 39 )
  {
    v21 = *(_QWORD *)(a3 - 48);
    if ( !v21 )
      goto LABEL_5;
    v22 = *(_QWORD *)(a3 - 24);
    if ( *(_BYTE *)(v22 + 16) != 13 )
      goto LABEL_5;
    goto LABEL_22;
  }
  if ( v6 != 5 )
  {
    if ( v6 != 47 )
      goto LABEL_5;
    v24 = *(_QWORD *)(a3 - 48);
    if ( !v24 )
      goto LABEL_5;
    v15 = *(_QWORD *)(a3 - 24);
    if ( *(_BYTE *)(v15 + 16) != 13 )
      goto LABEL_5;
    goto LABEL_10;
  }
  v7 = *(_WORD *)(a3 + 18);
  if ( v7 == 15 )
  {
    v21 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
    if ( !v21 )
      goto LABEL_5;
    v22 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    if ( *(_BYTE *)(v22 + 16) != 13 )
      goto LABEL_5;
LABEL_22:
    v25 = v22;
    v23 = sub_146F1B0(*(_QWORD *)(a1 + 176), a2);
    v12 = v25;
    v10 = (__int64)a4;
    v11 = v21;
    v13 = v23;
    return sub_1A69110(a1, 1, v13, v12, v11, v10);
  }
  if ( v7 != 23
    || (v24 = *(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))) == 0
    || (v15 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF))), *(_BYTE *)(v15 + 16) != 13) )
  {
LABEL_5:
    v8 = sub_159C470(*a4, 1, 0);
    v9 = sub_146F1B0(*(_QWORD *)(a1 + 176), a2);
    v10 = (__int64)a4;
    v11 = a3;
    v12 = v8;
    v13 = v9;
    return sub_1A69110(a1, 1, v13, v12, v11, v10);
  }
LABEL_10:
  v16 = *(_DWORD *)(v15 + 32);
  v27 = v16;
  if ( v16 <= 0x40 )
  {
    v29 = v16;
    v17 = v15 + 24;
    v26 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & 1;
LABEL_12:
    v28 = v26;
    goto LABEL_13;
  }
  sub_16A4EF0((__int64)&v26, 1, 0);
  v17 = v15 + 24;
  v29 = v27;
  if ( v27 <= 0x40 )
    goto LABEL_12;
  sub_16A4FD0((__int64)&v28, (const void **)&v26);
  v17 = v15 + 24;
LABEL_13:
  sub_16A7E20((__int64)&v28, v17);
  v18 = (__int64 *)sub_16498A0(v15);
  v19 = sub_159C0E0(v18, (__int64)&v28);
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  v20 = sub_146F1B0(*(_QWORD *)(a1 + 176), a2);
  result = sub_1A69110(a1, 1, v20, v19, v24, (__int64)a4);
  if ( v27 > 0x40 )
  {
    if ( v26 )
      return j_j___libc_free_0_0(v26);
  }
  return result;
}
