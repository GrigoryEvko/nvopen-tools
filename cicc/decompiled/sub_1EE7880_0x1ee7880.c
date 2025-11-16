// Function: sub_1EE7880
// Address: 0x1ee7880
//
void __fastcall sub_1EE7880(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rdx
  char v6; // r8
  int *v7; // rbx
  int *v8; // r12
  int v9; // esi
  unsigned int v10; // edi
  __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // r9
  _DWORD *v14; // rdx
  int v15; // r9d
  char *v16; // rax
  __int64 v17; // rdx
  char *v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // rdx
  char *v21; // rdx
  int v22; // eax
  int v23; // ecx
  int *v24; // rbx
  int *j; // r12
  int v26; // esi
  unsigned int v27; // edi
  __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // r9
  _DWORD *v31; // rdx
  int v32; // edx
  int v33; // ecx
  unsigned __int64 v34; // rdx
  __int64 i; // rcx
  __int64 v36; // r8
  unsigned int v37; // ecx
  unsigned int v38; // edi
  __int64 *v39; // rax
  __int64 v40; // r10
  int v41; // eax
  int v42; // r11d
  int *v43; // [rsp+10h] [rbp-120h] BYREF
  __int64 v44; // [rsp+18h] [rbp-118h]
  _BYTE v45[64]; // [rsp+20h] [rbp-110h] BYREF
  int *v46; // [rsp+60h] [rbp-D0h]
  __int64 v47; // [rsp+68h] [rbp-C8h]
  _BYTE v48[64]; // [rsp+70h] [rbp-C0h] BYREF
  int *v49; // [rsp+B0h] [rbp-80h]
  __int64 v50; // [rsp+B8h] [rbp-78h]
  _BYTE v51[112]; // [rsp+C0h] [rbp-70h] BYREF

  v3 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v34 = a2;
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
          (*(_BYTE *)(v34 + 46) & 4) != 0;
          v34 = *(_QWORD *)v34 & 0xFFFFFFFFFFFFFFF8LL )
    {
      ;
    }
    v36 = *(_QWORD *)(i + 368);
    v37 = *(_DWORD *)(i + 384);
    if ( v37 )
    {
      v38 = (v37 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v39 = (__int64 *)(v36 + 16LL * v38);
      v40 = *v39;
      if ( *v39 == v34 )
      {
LABEL_62:
        v3 = v39[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_2;
      }
      v41 = 1;
      while ( v40 != -8 )
      {
        v42 = v41 + 1;
        v38 = (v37 - 1) & (v41 + v38);
        v39 = (__int64 *)(v36 + 16LL * v38);
        v40 = *v39;
        if ( *v39 == v34 )
          goto LABEL_62;
        v41 = v42;
      }
    }
    v3 = *(_QWORD *)(v36 + 16LL * v37 + 8) & 0xFFFFFFFFFFFFFFF8LL | 4;
  }
LABEL_2:
  v4 = *(_QWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = *(_BYTE *)(a1 + 58);
  v43 = (int *)v45;
  v44 = 0x800000000LL;
  v46 = (int *)v48;
  v47 = 0x800000000LL;
  v49 = (int *)v51;
  v50 = 0x800000000LL;
  sub_1EE65F0((__int64)&v43, a2, v5, v4, v6, 1);
  if ( *(_BYTE *)(a1 + 58) )
  {
    sub_1EE6D60((__int64)&v43, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v3, 0);
  }
  else if ( *(_BYTE *)(a1 + 56) )
  {
    sub_1EE69C0((__int64)&v43, a2, *(_QWORD *)(a1 + 32));
  }
  sub_1EE7580(a1, v49, (unsigned int)v50);
  v7 = &v46[2 * (unsigned int)v47];
  v8 = v46;
  if ( v7 != v46 )
  {
    while ( 1 )
    {
      v9 = *v8;
      v10 = *v8;
      if ( *v8 < 0 )
        v10 = *(_DWORD *)(a1 + 192) + (v10 & 0x7FFFFFFF);
      v11 = *(unsigned int *)(a1 + 104);
      v12 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v10);
      if ( v12 >= (unsigned int)v11 )
        goto LABEL_42;
      v13 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v14 = (_DWORD *)(v13 + 8LL * v12);
        if ( v10 == *v14 )
          break;
        v12 += 256;
        if ( (unsigned int)v11 <= v12 )
          goto LABEL_42;
      }
      if ( v14 == (_DWORD *)(v13 + 8 * v11) )
LABEL_42:
        v15 = 0;
      else
        v15 = v14[1];
      v16 = (char *)v43;
      v17 = 8LL * (unsigned int)v44;
      v18 = (char *)&v43[(unsigned __int64)v17 / 4];
      v19 = v17 >> 3;
      v20 = v17 >> 5;
      if ( v20 )
      {
        v21 = (char *)&v43[8 * v20];
        while ( v9 != *(_DWORD *)v16 )
        {
          if ( v9 == *((_DWORD *)v16 + 2) )
          {
            v16 += 8;
            goto LABEL_21;
          }
          if ( v9 == *((_DWORD *)v16 + 4) )
          {
            v16 += 16;
            goto LABEL_21;
          }
          if ( v9 == *((_DWORD *)v16 + 6) )
          {
            v16 += 24;
            goto LABEL_21;
          }
          v16 += 32;
          if ( v21 == v16 )
          {
            v19 = (v18 - v16) >> 3;
            goto LABEL_44;
          }
        }
        goto LABEL_21;
      }
LABEL_44:
      if ( v19 == 2 )
        goto LABEL_50;
      if ( v19 == 3 )
        break;
      if ( v19 != 1 )
      {
LABEL_47:
        v22 = 0;
        goto LABEL_23;
      }
LABEL_52:
      if ( v9 != *(_DWORD *)v16 )
      {
        v22 = 0;
        goto LABEL_23;
      }
LABEL_21:
      if ( v18 == v16 )
        goto LABEL_47;
      v22 = *((_DWORD *)v16 + 1);
LABEL_23:
      v23 = v8[1];
      v8 += 2;
      sub_1EE5E20(a1, v9, v15, v22 | v15 & ~v23);
      if ( v7 == v8 )
        goto LABEL_24;
    }
    if ( v9 == *(_DWORD *)v16 )
      goto LABEL_21;
    v16 += 8;
LABEL_50:
    if ( v9 == *(_DWORD *)v16 )
      goto LABEL_21;
    v16 += 8;
    goto LABEL_52;
  }
LABEL_24:
  v24 = &v43[2 * (unsigned int)v44];
  for ( j = v43; v24 != j; j += 2 )
  {
    v26 = *j;
    v27 = *j;
    if ( *j < 0 )
      v27 = *(_DWORD *)(a1 + 192) + (v27 & 0x7FFFFFFF);
    v28 = *(unsigned int *)(a1 + 104);
    v29 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v27);
    if ( v29 >= (unsigned int)v28 )
      goto LABEL_41;
    v30 = *(_QWORD *)(a1 + 96);
    while ( 1 )
    {
      v31 = (_DWORD *)(v30 + 8LL * v29);
      if ( v27 == *v31 )
        break;
      v29 += 256;
      if ( (unsigned int)v28 <= v29 )
        goto LABEL_41;
    }
    if ( v31 == (_DWORD *)(v30 + 8 * v28) )
LABEL_41:
      v32 = 0;
    else
      v32 = v31[1];
    v33 = j[1];
    sub_1EE5D10(a1, v26, v32, v32 | v33);
  }
  if ( v49 != (int *)v51 )
    _libc_free((unsigned __int64)v49);
  if ( v46 != (int *)v48 )
    _libc_free((unsigned __int64)v46);
  if ( v43 != (int *)v45 )
    _libc_free((unsigned __int64)v43);
}
