// Function: sub_25C9E60
// Address: 0x25c9e60
//
char __fastcall sub_25C9E60(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // rax
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r9
  __int64 **v8; // rbx
  __int64 **i; // r13
  __int64 *v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r15
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rcx
  _BYTE *v20; // rdi
  __int64 v21; // rax
  int v22; // r15d
  unsigned __int8 *v23; // rbx
  unsigned __int8 v24; // al
  bool v25; // al
  char v26; // al
  __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 *v29; // rax
  __int64 *v30; // rbx
  __int64 *v31; // r8
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r9
  __int64 *v39; // [rsp+18h] [rbp-C8h]
  __int64 *v40; // [rsp+20h] [rbp-C0h]
  __int64 v41; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v42; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v43; // [rsp+48h] [rbp-98h]
  __int64 v44; // [rsp+50h] [rbp-90h]
  __int64 v45; // [rsp+58h] [rbp-88h]
  _BYTE *v46; // [rsp+60h] [rbp-80h]
  __int64 v47; // [rsp+68h] [rbp-78h]
  _BYTE v48[112]; // [rsp+70h] [rbp-70h] BYREF

  v2 = *(__int64 **)(a1 + 32);
  v3 = *(unsigned int *)(a1 + 40);
  v4 = &v2[v3];
  if ( v2 == v4 )
    return v3;
  while ( 1 )
  {
    while ( 1 )
    {
      v5 = *v2;
      if ( !(unsigned __int8)sub_A74710((_QWORD *)(*v2 + 120), 0, 22) )
      {
        LOBYTE(v3) = sub_B2FC80(v5);
        if ( (_BYTE)v3 )
          return v3;
        LOBYTE(v3) = sub_B2FC00((_BYTE *)v5);
        if ( (_BYTE)v3 )
          return v3;
        if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v5 + 24) + 16LL) + 8LL) == 14 )
          break;
      }
      if ( v4 == ++v2 )
        goto LABEL_8;
    }
    v46 = v48;
    v14 = v5 + 72;
    v42 = 0;
    v43 = 0;
    v44 = 0;
    v45 = 0;
    v47 = 0x800000000LL;
    v15 = *(_QWORD *)(v14 + 8);
    if ( v15 != v14 )
      break;
LABEL_32:
    ++v2;
    sub_C7D6A0(v43, 8LL * (unsigned int)v45, 8);
    if ( v4 == v2 )
    {
LABEL_8:
      v8 = *(__int64 ***)(a1 + 32);
      v3 = *(unsigned int *)(a1 + 40);
      for ( i = &v8[v3]; i != v8; ++v8 )
      {
        v10 = *v8;
        LOBYTE(v3) = sub_A74710(*v8 + 15, 0, 22);
        if ( !(_BYTE)v3 )
        {
          v3 = **(_QWORD **)(v10[3] + 16);
          if ( *(_BYTE *)(v3 + 8) == 14 )
          {
            sub_B2D390((__int64)v10, 22);
            LOBYTE(v3) = sub_25C0F40((__int64)&v42, a2, v10, v11, v12, v13);
          }
        }
      }
      return v3;
    }
  }
  do
  {
    if ( !v15 )
      BUG();
    v16 = *(_QWORD *)(v15 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v16 == v15 + 24 )
      goto LABEL_67;
    if ( !v16 )
      BUG();
    v17 = (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30;
    if ( (unsigned int)v17 > 0xA )
LABEL_67:
      BUG();
    if ( *(_BYTE *)(v16 - 24) == 30 )
    {
      v18 = 0;
      v19 = *(_DWORD *)(v16 - 20) & 0x7FFFFFF;
      if ( (*(_DWORD *)(v16 - 20) & 0x7FFFFFF) != 0 )
      {
        v19 = -32LL * (unsigned int)v19;
        v18 = *(_QWORD *)(v16 + v19 - 24);
      }
      v41 = v18;
      sub_25C9420((__int64)&v42, &v41, v6, v19, (__int64)&v41, v7);
    }
    v15 = *(_QWORD *)(v15 + 8);
  }
  while ( v14 != v15 );
  v20 = v46;
  if ( !(_DWORD)v47 )
  {
LABEL_30:
    if ( v20 != v48 )
      _libc_free((unsigned __int64)v20);
    goto LABEL_32;
  }
  v21 = 0;
  v22 = 0;
  while ( 1 )
  {
    v23 = *(unsigned __int8 **)&v20[8 * v21];
    v24 = *v23;
    if ( *v23 > 0x15u )
      break;
    v25 = sub_AC30F0((__int64)v23);
    v20 = v46;
    if ( !v25 && (unsigned int)*v23 - 12 > 1 )
      goto LABEL_39;
LABEL_29:
    v21 = (unsigned int)(v22 + 1);
    v22 = v21;
    if ( (_DWORD)v47 == (_DWORD)v21 )
      goto LABEL_30;
  }
  if ( v24 == 22 )
    goto LABEL_39;
  if ( v24 <= 0x1Cu )
  {
LABEL_38:
    v26 = sub_D13FA0((__int64)v23, 0, 0);
    v20 = v46;
    if ( v26 )
      goto LABEL_39;
    goto LABEL_29;
  }
  switch ( *v23 )
  {
    case '"':
    case 'U':
      if ( (unsigned __int8)sub_A74710((_QWORD *)v23 + 9, 0, 22) )
        goto LABEL_38;
      v28 = *((_QWORD *)v23 - 4);
      if ( !v28 || *(_BYTE *)v28 )
        goto LABEL_52;
      if ( *(_QWORD *)(v28 + 24) != *((_QWORD *)v23 + 10) )
        goto LABEL_51;
      v41 = *(_QWORD *)(v28 + 120);
      if ( (unsigned __int8)sub_A74710(&v41, 0, 22) )
        goto LABEL_38;
      v28 = *((_QWORD *)v23 - 4);
      if ( !v28 || *(_BYTE *)v28 )
        goto LABEL_52;
LABEL_51:
      if ( *(_QWORD *)(v28 + 24) == *((_QWORD *)v23 + 10) )
      {
        v41 = v28;
        if ( sub_25C04C0(a1, &v41) )
          goto LABEL_38;
      }
LABEL_52:
      v20 = v46;
      break;
    case '<':
      goto LABEL_38;
    case '?':
    case 'N':
    case 'O':
      if ( (v23[7] & 0x40) != 0 )
        v27 = (__int64 *)*((_QWORD *)v23 - 1);
      else
        v27 = (__int64 *)&v23[-32 * (*((_DWORD *)v23 + 1) & 0x7FFFFFF)];
      v41 = *v27;
      sub_25C9420((__int64)&v42, &v41, v6, v17, (__int64)&v41, v7);
      v20 = v46;
      goto LABEL_29;
    case 'T':
      v6 = 32LL * (*((_DWORD *)v23 + 1) & 0x7FFFFFF);
      if ( (v23[7] & 0x40) != 0 )
      {
        v29 = (__int64 *)*((_QWORD *)v23 - 1);
        v6 += (__int64)v29;
        v39 = (__int64 *)v6;
      }
      else
      {
        v39 = (__int64 *)v23;
        v29 = (__int64 *)&v23[-v6];
      }
      if ( v29 != v39 )
      {
        v30 = v29;
        v31 = &v41;
        do
        {
          v32 = *v30;
          v30 += 4;
          v40 = v31;
          v41 = v32;
          sub_25C9420((__int64)&v42, v31, v32, v17, (__int64)v31, v7);
          v31 = v40;
        }
        while ( v39 != v30 );
        v20 = v46;
      }
      goto LABEL_29;
    case 'V':
      v41 = *((_QWORD *)v23 - 8);
      sub_25C9420((__int64)&v42, &v41, v6, v17, (__int64)&v41, v7);
      v41 = *((_QWORD *)v23 - 4);
      sub_25C9420((__int64)&v42, &v41, v33, v34, (__int64)&v41, v35);
      v20 = v46;
      goto LABEL_29;
    default:
      if ( v20 != v48 )
        goto LABEL_40;
      goto LABEL_41;
  }
LABEL_39:
  if ( v20 != v48 )
LABEL_40:
    _libc_free((unsigned __int64)v20);
LABEL_41:
  LOBYTE(v3) = sub_C7D6A0(v43, 8LL * (unsigned int)v45, 8);
  return v3;
}
