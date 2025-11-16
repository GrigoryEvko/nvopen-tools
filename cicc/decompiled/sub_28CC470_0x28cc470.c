// Function: sub_28CC470
// Address: 0x28cc470
//
__int64 __fastcall sub_28CC470(__int64 a1, _BYTE *a2, __int64 a3)
{
  int v3; // r14d
  __int64 *v4; // rax
  int v5; // r15d
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *v8; // rdx
  _BYTE *v10; // r15
  signed __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // rax
  bool v14; // cf
  unsigned __int64 v15; // rax
  char *v16; // r14
  char *v17; // rcx
  __int64 v18; // r8
  char *v19; // rax
  unsigned __int64 v20; // r14
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  char *v24; // [rsp+8h] [rbp-48h]
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = -1;
  if ( a2 )
  {
    v3 = 0;
    if ( *a2 > 0x1Cu )
    {
      v25[0] = (__int64)a2;
      v4 = sub_28CBE90(a1 + 2416, v25);
      if ( v4 )
        v3 = *((_DWORD *)v4 + 2);
    }
  }
  v5 = *(_DWORD *)(a1 + 1424);
  *(_DWORD *)(a1 + 1424) = v5 + 1;
  v6 = sub_22077B0(0xB8u);
  v7 = v6;
  if ( v6 )
  {
    *(_DWORD *)v6 = v5;
    *(_QWORD *)(v6 + 8) = a2;
    *(_DWORD *)(v6 + 16) = v3;
    *(_QWORD *)(v6 + 24) = 0;
    *(_DWORD *)(v6 + 32) = -1;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    *(_QWORD *)(v6 + 64) = 0;
    *(_QWORD *)(v6 + 56) = a3;
    *(_QWORD *)(v6 + 72) = v6 + 96;
    *(_QWORD *)(v6 + 80) = 4;
    *(_DWORD *)(v6 + 88) = 0;
    *(_BYTE *)(v6 + 92) = 1;
    *(_QWORD *)(v6 + 128) = 0;
    *(_QWORD *)(v6 + 136) = v6 + 160;
    *(_QWORD *)(v6 + 144) = 2;
    *(_DWORD *)(v6 + 152) = 0;
    *(_BYTE *)(v6 + 156) = 1;
    *(_DWORD *)(v6 + 176) = 0;
  }
  v8 = *(__int64 **)(a1 + 1408);
  if ( v8 == *(__int64 **)(a1 + 1416) )
  {
    v10 = *(_BYTE **)(a1 + 1400);
    v11 = (char *)v8 - v10;
    v12 = ((char *)v8 - v10) >> 3;
    if ( v12 == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"vector::_M_realloc_insert");
    v13 = 1;
    if ( v12 )
      v13 = ((char *)v8 - v10) >> 3;
    v14 = __CFADD__(v12, v13);
    v15 = v12 + v13;
    if ( v14 )
    {
      v20 = 0x7FFFFFFFFFFFFFF8LL;
    }
    else
    {
      if ( !v15 )
      {
        v16 = 0;
        v17 = 0;
        goto LABEL_18;
      }
      if ( v15 > 0xFFFFFFFFFFFFFFFLL )
        v15 = 0xFFFFFFFFFFFFFFFLL;
      v20 = 8 * v15;
    }
    v17 = (char *)sub_22077B0(v20);
    v16 = &v17[v20];
LABEL_18:
    if ( &v17[v11] )
      *(_QWORD *)&v17[v11] = v7;
    v18 = (__int64)&v17[v11 + 8];
    if ( v11 > 0 )
    {
      v23 = (__int64)&v17[v11 + 8];
      v19 = (char *)memmove(v17, v10, v11);
      v18 = v23;
      v17 = v19;
    }
    else if ( !v10 )
    {
LABEL_22:
      *(_QWORD *)(a1 + 1400) = v17;
      *(_QWORD *)(a1 + 1408) = v18;
      *(_QWORD *)(a1 + 1416) = v16;
      return v7;
    }
    v21 = v18;
    v24 = v17;
    j_j___libc_free_0((unsigned __int64)v10);
    v18 = v21;
    v17 = v24;
    goto LABEL_22;
  }
  if ( v8 )
  {
    *v8 = v6;
    v8 = *(__int64 **)(a1 + 1408);
  }
  *(_QWORD *)(a1 + 1408) = v8 + 1;
  return v7;
}
