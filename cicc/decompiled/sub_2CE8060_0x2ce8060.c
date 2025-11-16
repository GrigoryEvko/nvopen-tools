// Function: sub_2CE8060
// Address: 0x2ce8060
//
__int64 __fastcall sub_2CE8060(__int64 a1)
{
  unsigned __int64 v2; // r12
  _QWORD *v3; // rdi
  _QWORD *v4; // rax
  unsigned int v5; // r13d
  __int64 v6; // r12
  char v7; // al
  __int64 *v8; // rdx
  __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 *v11; // rbx
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 *v15; // r12
  __int64 v16; // rax
  __int64 v17; // [rsp+8h] [rbp-78h] BYREF
  unsigned __int64 v18[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v19; // [rsp+20h] [rbp-60h]
  _QWORD *v20; // [rsp+28h] [rbp-58h]
  _QWORD *v21; // [rsp+30h] [rbp-50h]
  unsigned __int64 v22; // [rsp+38h] [rbp-48h]
  _QWORD *v23; // [rsp+40h] [rbp-40h]
  _QWORD *v24; // [rsp+48h] [rbp-38h]
  _QWORD *v25; // [rsp+50h] [rbp-30h]
  unsigned __int64 *v26; // [rsp+58h] [rbp-28h]

  v23 = 0;
  v18[1] = 8;
  v18[0] = sub_22077B0(0x40u);
  v2 = v18[0] + 24;
  v22 = v18[0] + 24;
  v3 = (_QWORD *)sub_22077B0(0x200u);
  *(_QWORD *)(v18[0] + 24) = v3;
  v20 = v3;
  v21 = v3 + 64;
  v26 = (unsigned __int64 *)v2;
  v24 = v3;
  v25 = v3 + 64;
  v19 = v3;
  if ( v3 )
    *v3 = a1;
  v4 = v3 + 1;
  v5 = 0;
  v23 = v3 + 1;
  while ( 1 )
  {
    v6 = *(v4 - 1);
    v23 = v4 - 1;
    v7 = *(_BYTE *)v6;
    if ( *(_BYTE *)v6 > 0x1Cu )
      break;
    while ( 1 )
    {
      if ( v7 != 17 )
      {
        if ( v7 != 22 )
          goto LABEL_20;
        v5 = sub_B2D680(v6) ^ 1;
      }
LABEL_12:
      v4 = v23;
      if ( v23 == v19 )
        goto LABEL_21;
      if ( v24 != v23 )
        break;
      v6 = *(_QWORD *)(*(v26 - 1) + 504);
      j_j___libc_free_0((unsigned __int64)v24);
      v24 = (_QWORD *)*--v26;
      v25 = v24 + 64;
      v23 = v24 + 63;
      v7 = *(_BYTE *)v6;
      if ( *(_BYTE *)v6 > 0x1Cu )
        goto LABEL_5;
    }
  }
LABEL_5:
  if ( v7 == 93 )
    goto LABEL_29;
  if ( (unsigned __int8)(v7 - 54) <= 4u )
  {
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
      v8 = *(__int64 **)(v6 - 8);
    else
      v8 = (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
    v17 = *v8;
    sub_2CE7F40(v18, &v17);
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
      v9 = *(_QWORD *)(v6 - 8);
    else
      v9 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
    v17 = *(_QWORD *)(v9 + 32);
    sub_2CE7F40(v18, &v17);
    goto LABEL_12;
  }
  if ( (unsigned __int8)(v7 - 67) <= 1u )
  {
LABEL_29:
    if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
      v15 = *(__int64 **)(v6 - 8);
    else
      v15 = (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
    v17 = *v15;
    sub_2CE7F40(v18, &v17);
    goto LABEL_12;
  }
  if ( v7 == 61 )
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v6 - 32) + 8LL);
    if ( *(_BYTE *)(v16 + 8) == 14 && *(_DWORD *)(v16 + 8) >> 8 == 101 )
    {
      v5 = 1;
      goto LABEL_12;
    }
  }
LABEL_20:
  v5 = 0;
LABEL_21:
  v10 = v18[0];
  if ( v18[0] )
  {
    v11 = (unsigned __int64 *)v22;
    v12 = v26 + 1;
    if ( (unsigned __int64)(v26 + 1) > v22 )
    {
      do
      {
        v13 = *v11++;
        j_j___libc_free_0(v13);
      }
      while ( v12 > v11 );
      v10 = v18[0];
    }
    j_j___libc_free_0(v10);
  }
  return v5;
}
