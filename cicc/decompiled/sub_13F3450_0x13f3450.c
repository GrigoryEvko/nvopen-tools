// Function: sub_13F3450
// Address: 0x13f3450
//
__int64 __fastcall sub_13F3450(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r8
  unsigned __int8 v11; // al
  __int64 v12; // r15
  unsigned __int8 v13; // al
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 *v17; // r12
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rdx
  int v25; // eax
  __int64 v26; // [rsp+0h] [rbp-C0h]
  unsigned int v27; // [rsp+Ch] [rbp-B4h]
  int v28; // [rsp+10h] [rbp-B0h]
  __int64 v29; // [rsp+10h] [rbp-B0h]
  unsigned int v30; // [rsp+18h] [rbp-A8h]
  __int64 v31; // [rsp+20h] [rbp-A0h]
  __int64 v32; // [rsp+20h] [rbp-A0h]
  int v34; // [rsp+30h] [rbp-90h] BYREF
  __int64 v35; // [rsp+38h] [rbp-88h]
  unsigned int v36; // [rsp+40h] [rbp-80h]
  __int64 v37; // [rsp+48h] [rbp-78h]
  unsigned int v38; // [rsp+50h] [rbp-70h]
  int v39; // [rsp+60h] [rbp-60h] BYREF
  __int64 v40; // [rsp+68h] [rbp-58h]
  unsigned int v41; // [rsp+70h] [rbp-50h]
  __int64 v42; // [rsp+78h] [rbp-48h]
  unsigned int v43; // [rsp+80h] [rbp-40h]

  v7 = a1;
  v8 = sub_15F2050(a5);
  v9 = sub_1632FA0(v8);
  if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 15 )
  {
    if ( (unsigned __int8)sub_1593BB0(a4) )
    {
      v22 = sub_1649C60(a3);
      if ( (unsigned __int8)sub_14BFF20(v22, v9, 0, 0, 0, 0) )
      {
        v30 = 0;
        if ( a2 == 32 )
          return v30;
        v30 = 1;
        if ( a2 == 33 )
          return v30;
      }
    }
  }
  v10 = sub_13E7A30(a1 + 4, *a1, v9, a1[3]);
  v11 = *(_BYTE *)(a3 + 16);
  if ( v11 > 0x10u )
  {
    v34 = 4;
    if ( v11 > 0x17u )
    {
      v29 = v10;
      sub_13EA4E0(&v39, a3);
      sub_13E8810(&v34, (unsigned int *)&v39);
      v10 = v29;
      if ( v39 == 3 )
      {
        if ( v43 > 0x40 && v42 )
        {
          j_j___libc_free_0_0(v42);
          v10 = v29;
        }
        if ( v41 > 0x40 && v40 )
        {
          v32 = v10;
          j_j___libc_free_0_0(v40);
          v10 = v32;
        }
      }
    }
    sub_13EE9C0(v10, a3, &v34, a5);
    v39 = 0;
    sub_13E8810(&v39, (unsigned int *)&v34);
    if ( v34 == 3 )
    {
      if ( v38 > 0x40 && v37 )
        j_j___libc_free_0_0(v37);
      if ( v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
    }
  }
  else
  {
    v39 = 0;
    if ( v11 != 9 )
      sub_13EA740(&v39, a3);
  }
  v30 = sub_13E9B70(a2, a4, &v39, v9, a1[2]);
  if ( v30 != -1 || !a5 )
    goto LABEL_17;
  v12 = *(_QWORD *)(*(_QWORD *)(a5 + 40) + 8LL);
  v31 = *(_QWORD *)(a5 + 40);
  if ( !v12 )
  {
LABEL_20:
    v28 = -1;
LABEL_21:
    v30 = v28;
    goto LABEL_17;
  }
  while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v12) + 16) - 25) > 9u )
  {
    v12 = *(_QWORD *)(v12 + 8);
    if ( !v12 )
      goto LABEL_20;
  }
  v13 = *(_BYTE *)(a3 + 16);
  if ( v13 != 77 )
    goto LABEL_10;
  if ( *(_QWORD *)(a3 + 40) != v31 )
  {
LABEL_12:
    v14 = sub_1648700(v12);
    v28 = sub_13F3340(v7, a2, a3, a4, *(_QWORD *)(v14 + 40), v31, a5);
    if ( v28 == -1 )
      goto LABEL_17;
    v15 = a4;
    v16 = a3;
    v17 = v7;
    v18 = v12;
    v19 = v15;
    while ( 1 )
    {
      v18 = *(_QWORD *)(v18 + 8);
      if ( !v18 )
        goto LABEL_21;
      v20 = sub_1648700(v18);
      if ( (unsigned __int8)(*(_BYTE *)(v20 + 16) - 25) <= 9u
        && v28 != (unsigned int)sub_13F3340(v17, a2, v16, v19, *(_QWORD *)(v20 + 40), v31, a5) )
      {
        goto LABEL_17;
      }
    }
  }
  v27 = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( v27 )
  {
    v26 = v12;
    v28 = -1;
    v23 = 0;
    while ( 1 )
    {
      v24 = (*(_BYTE *)(a3 + 23) & 0x40) != 0 ? *(_QWORD *)(a3 - 8) : a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
      v25 = sub_13F3340(
              a1,
              a2,
              *(_QWORD *)(v24 + 24 * v23),
              a4,
              *(_QWORD *)(v24 + 8 * v23 + 24LL * *(unsigned int *)(a3 + 56) + 8),
              v31,
              a5);
      if ( v23 )
      {
        if ( v28 != v25 )
          break;
      }
      if ( v25 == -1 )
        break;
      ++v23;
      v28 = v25;
      if ( v27 <= (unsigned int)v23 )
        goto LABEL_21;
    }
    v12 = v26;
    v7 = a1;
    v13 = *(_BYTE *)(a3 + 16);
LABEL_10:
    if ( v13 <= 0x17u || v31 != *(_QWORD *)(a3 + 40) )
      goto LABEL_12;
  }
LABEL_17:
  if ( v39 == 3 )
  {
    if ( v43 > 0x40 && v42 )
      j_j___libc_free_0_0(v42);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
  }
  return v30;
}
