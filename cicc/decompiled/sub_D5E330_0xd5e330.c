// Function: sub_D5E330
// Address: 0xd5e330
//
__int64 __fastcall sub_D5E330(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  int v7; // edx
  __int64 v9; // r13
  char v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  const void *v13; // rax
  unsigned int v14; // edx
  __int16 v15; // ax
  char v16; // si
  unsigned __int16 v17; // cx
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned int v20; // eax
  unsigned int v21; // eax
  char v22; // [rsp+7h] [rbp-79h]
  __int64 v23; // [rsp+8h] [rbp-78h]
  unsigned __int16 v24; // [rsp+8h] [rbp-78h]
  const void *v25; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v26; // [rsp+18h] [rbp-68h]
  const void *v27; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-58h]
  const void *v29; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-48h]
  unsigned __int64 v31; // [rsp+40h] [rbp-40h] BYREF
  __int64 v32; // [rsp+48h] [rbp-38h]

  v6 = *(_QWORD *)(a3 + 24);
  v7 = *(unsigned __int8 *)(v6 + 8);
  if ( (_BYTE)v7 != 12
    && (unsigned __int8)v7 > 3u
    && (_BYTE)v7 != 5
    && (v7 & 0xFD) != 4
    && (v7 & 0xFB) != 0xA
    && ((unsigned __int8)(v7 - 15) > 3u && v7 != 20 || !(unsigned __int8)sub_BCEBA0(v6, 0))
    || (*(_BYTE *)(a3 + 32) & 0xF) == 9
    || (sub_B2FC80(a3) || (unsigned __int8)sub_B2F6B0(a3)) && *(_BYTE *)(a2 + 16) != 2 )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v9 = *(_QWORD *)a2;
  v23 = *(_QWORD *)(a3 + 24);
  v10 = sub_AE5020(*(_QWORD *)a2, v23);
  v11 = sub_9208B0(v9, v23);
  v32 = v12;
  v31 = ((1LL << v10) + ((unsigned __int64)(v11 + 7) >> 3) - 1) >> v10 << v10;
  v13 = (const void *)sub_CA1930(&v31);
  v14 = *(_DWORD *)(a2 + 32);
  v26 = v14;
  if ( v14 <= 0x40 )
  {
    v25 = v13;
    v15 = (*(_WORD *)(a3 + 34) >> 1) & 0x3F;
    if ( !v15 )
    {
      v17 = 0;
      v28 = v14;
      goto LABEL_19;
    }
    goto LABEL_17;
  }
  sub_C43690((__int64)&v25, (__int64)v13, 0);
  v14 = v26;
  v16 = 0;
  v15 = (*(_WORD *)(a3 + 34) >> 1) & 0x3F;
  if ( v15 )
  {
LABEL_17:
    v16 = 1;
    v22 = v15 - 1;
  }
  LOBYTE(v17) = v22;
  v28 = v14;
  HIBYTE(v17) = v16;
  if ( v14 <= 0x40 )
  {
LABEL_19:
    v27 = v25;
    goto LABEL_20;
  }
  v24 = v17;
  sub_C43780((__int64)&v27, &v25);
  v17 = v24;
LABEL_20:
  sub_D5D630((__int64)&v29, a2, &v27, v17);
  v18 = *(_DWORD *)(a2 + 48);
  LODWORD(v32) = v18;
  if ( v18 > 0x40 )
  {
    sub_C43780((__int64)&v31, (const void **)(a2 + 40));
    v21 = v32;
    *(_DWORD *)(a1 + 8) = v32;
    if ( v21 > 0x40 )
    {
      sub_C43780(a1, (const void **)&v31);
      goto LABEL_23;
    }
  }
  else
  {
    v19 = *(_QWORD *)(a2 + 40);
    *(_DWORD *)(a1 + 8) = v18;
    v31 = v19;
  }
  *(_QWORD *)a1 = v31;
LABEL_23:
  v20 = v30;
  *(_DWORD *)(a1 + 24) = v30;
  if ( v20 > 0x40 )
    sub_C43780(a1 + 16, &v29);
  else
    *(_QWORD *)(a1 + 16) = v29;
  if ( (unsigned int)v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  if ( v28 > 0x40 && v27 )
    j_j___libc_free_0_0(v27);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return a1;
}
