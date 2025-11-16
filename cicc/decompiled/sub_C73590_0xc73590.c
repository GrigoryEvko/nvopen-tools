// Function: sub_C73590
// Address: 0xc73590
//
__int64 __fastcall sub_C73590(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // ebx
  unsigned __int64 v6; // r13
  __int64 v7; // r13
  char v8; // cl
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rbx
  unsigned int v11; // r13d
  __int64 v12; // rax
  int v13; // edx
  char v14; // r15
  unsigned int v15; // ebx
  unsigned int v16; // edx
  unsigned int v17; // esi
  __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // r13
  __int64 v21; // r13
  unsigned int v22; // eax
  unsigned __int64 v23; // rdx
  int v25; // edx
  unsigned int v26; // eax
  bool v27; // cc
  __int64 v28; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+8h] [rbp-58h]
  __int64 v30; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v33; // [rsp+28h] [rbp-38h]

  v5 = *(_DWORD *)(a2 + 8);
  v31 = v5;
  if ( v5 <= 0x40 )
  {
    v6 = *(_QWORD *)a2;
LABEL_3:
    v7 = *(_QWORD *)a3 | v6;
    v31 = 0;
    v30 = v7;
LABEL_4:
    if ( v5 )
    {
      v8 = 64 - v5;
      v5 = 64;
      v9 = ~(v7 << v8);
      if ( v9 )
      {
        _BitScanReverse64(&v10, v9);
        v5 = v10 ^ 0x3F;
      }
    }
    goto LABEL_7;
  }
  sub_C43780((__int64)&v30, (const void **)a2);
  v5 = v31;
  if ( v31 <= 0x40 )
  {
    v6 = v30;
    goto LABEL_3;
  }
  sub_C43BD0(&v30, (__int64 *)a3);
  v5 = v31;
  v7 = v30;
  v31 = 0;
  v33 = v5;
  v32 = v30;
  if ( v5 <= 0x40 )
    goto LABEL_4;
  v5 = sub_C44500((__int64)&v32);
  if ( v7 )
  {
    j_j___libc_free_0_0(v7);
    if ( v31 > 0x40 )
    {
      if ( v30 )
      {
        j_j___libc_free_0_0(v30);
        v11 = *(_DWORD *)(a3 + 8);
        v29 = v11;
        if ( v11 <= 0x40 )
          goto LABEL_8;
        goto LABEL_36;
      }
    }
  }
LABEL_7:
  v11 = *(_DWORD *)(a3 + 8);
  v29 = v11;
  if ( v11 <= 0x40 )
  {
LABEL_8:
    v12 = *(_QWORD *)a3;
    v13 = *(_DWORD *)(a2 + 8);
    v33 = v11;
    v28 = v12;
    v14 = v13 - v5;
    v15 = v11 - v13 + v5;
    goto LABEL_9;
  }
LABEL_36:
  sub_C43780((__int64)&v28, (const void **)a3);
  v11 = v29;
  v25 = *(_DWORD *)(a2 + 8);
  v33 = v29;
  v14 = v25 - v5;
  v15 = v29 - v25 + v5;
  if ( v29 <= 0x40 )
  {
LABEL_9:
    v32 = 0;
    v16 = v11;
    v17 = v11 - v15;
    if ( v11 == v11 - v15 )
    {
      v18 = 0;
      goto LABEL_15;
    }
    if ( v17 > 0x3F )
      goto LABEL_39;
    goto LABEL_11;
  }
  sub_C43690((__int64)&v32, 0, 0);
  v16 = v33;
  v17 = v33 - v15;
  if ( v33 != v33 - v15 )
  {
    if ( v17 > 0x3F )
      goto LABEL_39;
LABEL_11:
    if ( v16 <= 0x40 )
    {
      v32 |= 0xFFFFFFFFFFFFFFFFLL >> (v14 + 64 - (unsigned __int8)v11) << v17;
      goto LABEL_13;
    }
LABEL_39:
    sub_C43C90(&v32, v17, v16);
    if ( v29 <= 0x40 )
      goto LABEL_14;
LABEL_40:
    sub_C43B90(&v28, &v32);
    v17 = v33;
    goto LABEL_16;
  }
LABEL_13:
  if ( v29 > 0x40 )
    goto LABEL_40;
LABEL_14:
  v18 = v32;
  v17 = v33;
LABEL_15:
  v28 &= v18;
LABEL_16:
  if ( v17 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  v19 = *(_DWORD *)(a2 + 24);
  v31 = v19;
  if ( v19 <= 0x40 )
  {
    v20 = *(_QWORD *)(a2 + 16);
LABEL_21:
    v21 = v28 | v20;
    v30 = v21;
    goto LABEL_22;
  }
  sub_C43780((__int64)&v30, (const void **)(a2 + 16));
  v19 = v31;
  if ( v31 <= 0x40 )
  {
    v20 = v30;
    goto LABEL_21;
  }
  sub_C43BD0(&v30, &v28);
  v19 = v31;
  v21 = v30;
LABEL_22:
  v22 = *(_DWORD *)(a2 + 8);
  v31 = 0;
  v33 = v22;
  if ( v22 > 0x40 )
  {
    sub_C43780((__int64)&v32, (const void **)a2);
    v26 = v33;
    v27 = v31 <= 0x40;
    *(_DWORD *)(a1 + 24) = v19;
    *(_QWORD *)(a1 + 16) = v21;
    *(_DWORD *)(a1 + 8) = v26;
    *(_QWORD *)a1 = v32;
    if ( !v27 && v30 )
      j_j___libc_free_0_0(v30);
  }
  else
  {
    v23 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = v22;
    *(_DWORD *)(a1 + 24) = v19;
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 16) = v21;
  }
  if ( v29 > 0x40 && v28 )
    j_j___libc_free_0_0(v28);
  return a1;
}
