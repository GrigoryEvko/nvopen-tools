// Function: sub_3266230
// Address: 0x3266230
//
__int64 __fastcall sub_3266230(__int64 a1, __int64 a2)
{
  unsigned __int16 *v4; // rdx
  unsigned __int16 v5; // ax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // r13d
  unsigned __int16 *v10; // rdx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  void *v20; // rdi
  __int64 v21; // rax
  unsigned int v22; // edx
  unsigned int v23; // ecx
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-50h] BYREF
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-38h]
  __int64 v32; // [rsp+20h] [rbp-30h] BYREF
  __int64 v33; // [rsp+28h] [rbp-28h]

  v4 = *(unsigned __int16 **)(*(_QWORD *)(a2 + 8) + 48LL);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  LOWORD(v32) = v5;
  v33 = v6;
  if ( v5 )
  {
    if ( v5 == 1 || (unsigned __int16)(v5 - 504) <= 7u )
      goto LABEL_30;
    v8 = 16LL * (v5 - 1);
    v7 = *(_QWORD *)&byte_444C4A0[v8];
    LOBYTE(v8) = byte_444C4A0[v8 + 8];
  }
  else
  {
    v7 = sub_3007260((__int64)&v32);
    v30 = v7;
    v31 = v8;
  }
  LOBYTE(v33) = v8;
  v32 = v7;
  v9 = sub_CA1930(&v32);
  v10 = *(unsigned __int16 **)(*(_QWORD *)a2 + 48LL);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v28) = v11;
  v29 = v12;
  if ( v11 )
  {
    if ( v11 != 1 && (unsigned __int16)(v11 - 504) > 7u )
    {
      v27 = 16LL * (v11 - 1);
      v14 = *(_QWORD *)&byte_444C4A0[v27];
      v15 = byte_444C4A0[v27 + 8];
      goto LABEL_5;
    }
LABEL_30:
    BUG();
  }
  v32 = sub_3007260((__int64)&v28);
  v33 = v13;
  v14 = v32;
  v15 = v33;
LABEL_5:
  v28 = v14;
  LOBYTE(v29) = v15;
  v16 = (unsigned int)sub_CA1930(&v28);
  *(_DWORD *)(a1 + 8) = v16;
  if ( (unsigned int)v16 <= 0x40 || (sub_C43690(a1, 0, 0), v16 = *(unsigned int *)(a1 + 8), (unsigned int)v16 <= 0x40) )
  {
    *(_QWORD *)a1 = -1;
    v17 = -1;
  }
  else
  {
    memset(*(void **)a1, -1, 8 * (((unsigned __int64)(unsigned int)v16 + 63) >> 6));
    v16 = *(unsigned int *)(a1 + 8);
    v17 = *(_QWORD *)a1;
  }
  v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v16;
  if ( !(_DWORD)v16 )
  {
    v18 = 0;
    goto LABEL_20;
  }
  if ( (unsigned int)v16 <= 0x40 )
  {
LABEL_20:
    *(_QWORD *)a1 = v17 & v18;
    goto LABEL_10;
  }
  v19 = (unsigned int)((unsigned __int64)(v16 + 63) >> 6) - 1;
  *(_QWORD *)(v17 + 8 * v19) &= v18;
LABEL_10:
  sub_C449B0((__int64)&v28, (const void **)a1, v9);
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    v20 = *(void **)a1;
    if ( *(_QWORD *)a1 )
      j_j___libc_free_0_0((unsigned __int64)v20);
  }
  v21 = v28;
  v22 = v29;
  *(_QWORD *)a1 = v28;
  *(_DWORD *)(a1 + 8) = v22;
  v23 = *(_DWORD *)(a2 + 16);
  if ( v22 > 0x40 )
  {
    sub_C47690((__int64 *)a1, v23);
    return a1;
  }
  else
  {
    v24 = v21 << v23;
    if ( v23 == v22 )
      v24 = 0;
    v25 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v22) & v24;
    if ( !v22 )
      v25 = 0;
    *(_QWORD *)a1 = v25;
    return a1;
  }
}
