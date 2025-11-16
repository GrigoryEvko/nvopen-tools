// Function: sub_1C9F350
// Address: 0x1c9f350
//
__int64 __fastcall sub_1C9F350(__int64 a1)
{
  _QWORD *v2; // r12
  _QWORD *v3; // rdi
  _QWORD *v4; // rax
  unsigned int v5; // r13d
  __int64 v6; // r12
  unsigned __int8 v7; // al
  __int64 *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 *v11; // rbx
  __int64 *v12; // r12
  __int64 v13; // rdi
  __int64 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-78h] BYREF
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  _QWORD *v20; // [rsp+20h] [rbp-60h]
  _QWORD *v21; // [rsp+28h] [rbp-58h]
  _QWORD *v22; // [rsp+30h] [rbp-50h]
  unsigned __int64 v23; // [rsp+38h] [rbp-48h]
  _QWORD *v24; // [rsp+40h] [rbp-40h]
  _QWORD *v25; // [rsp+48h] [rbp-38h]
  _QWORD *v26; // [rsp+50h] [rbp-30h]
  _QWORD *v27; // [rsp+58h] [rbp-28h]

  v24 = 0;
  v19 = 8;
  v18 = sub_22077B0(64);
  v2 = (_QWORD *)(v18 + 24);
  v23 = v18 + 24;
  v3 = (_QWORD *)sub_22077B0(512);
  *(_QWORD *)(v18 + 24) = v3;
  v21 = v3;
  v22 = v3 + 64;
  v27 = v2;
  v25 = v3;
  v26 = v3 + 64;
  v20 = v3;
  if ( v3 )
    *v3 = a1;
  v4 = v3 + 1;
  v5 = 0;
  v24 = v3 + 1;
  while ( 1 )
  {
    v6 = *(v4 - 1);
    v24 = v4 - 1;
    v7 = *(_BYTE *)(v6 + 16);
    if ( v7 > 0x17u )
      break;
    while ( 1 )
    {
      if ( v7 != 13 )
      {
        if ( v7 != 17 )
          goto LABEL_20;
        v5 = sub_15E0450(v6) ^ 1;
      }
LABEL_12:
      v4 = v24;
      if ( v24 == v20 )
        goto LABEL_21;
      if ( v25 != v24 )
        break;
      v6 = *(_QWORD *)(*(v27 - 1) + 504LL);
      j_j___libc_free_0(v25, 512);
      v25 = (_QWORD *)*--v27;
      v26 = v25 + 64;
      v24 = v25 + 63;
      v7 = *(_BYTE *)(v6 + 16);
      if ( v7 > 0x17u )
        goto LABEL_5;
    }
  }
LABEL_5:
  if ( v7 == 86 )
    goto LABEL_29;
  if ( (unsigned __int8)(v7 - 47) <= 4u )
  {
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      v8 = *(__int64 **)(v6 - 8);
    else
      v8 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v17 = *v8;
    sub_1C70060(&v18, &v17);
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      v9 = *(_QWORD *)(v6 - 8);
    else
      v9 = v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
    v17 = *(_QWORD *)(v9 + 24);
    sub_1C70060(&v18, &v17);
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v7 - 60) <= 1u )
  {
LABEL_29:
    if ( (*(_BYTE *)(v6 + 23) & 0x40) != 0 )
      v15 = *(__int64 **)(v6 - 8);
    else
      v15 = (__int64 *)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
    v17 = *v15;
    sub_1C70060(&v18, &v17);
    goto LABEL_12;
  }
  if ( v7 == 54 )
  {
    v16 = **(_QWORD **)(v6 - 24);
    if ( *(_BYTE *)(v16 + 8) == 15 && *(_DWORD *)(v16 + 8) >> 8 == 101 )
    {
      v5 = 1;
      goto LABEL_12;
    }
  }
LABEL_20:
  v5 = 0;
LABEL_21:
  v10 = v18;
  if ( v18 )
  {
    v11 = (__int64 *)v23;
    v12 = v27 + 1;
    if ( (unsigned __int64)(v27 + 1) > v23 )
    {
      do
      {
        v13 = *v11++;
        j_j___libc_free_0(v13, 512);
      }
      while ( v12 > v11 );
      v10 = v18;
    }
    j_j___libc_free_0(v10, 8 * v19);
  }
  return v5;
}
