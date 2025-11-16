// Function: sub_D1C200
// Address: 0xd1c200
//
_BOOL8 __fastcall sub_D1C200(__int64 a1, unsigned __int8 **a2, unsigned __int8 **a3, __int64 a4, __int64 a5)
{
  unsigned __int8 *v7; // rax
  unsigned __int8 *v8; // rbx
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // r12
  unsigned __int8 v11; // al
  __int64 v12; // r15
  __int64 **v13; // rax
  __int64 **v14; // rcx
  __int64 **v15; // rdx
  __int64 *v16; // r8
  __int64 v18; // rdx
  unsigned __int8 *v19; // r15
  unsigned __int8 *v20; // r13
  unsigned __int8 **v21; // rax
  unsigned __int8 **v22; // rdx
  int v23; // edx
  __int64 v24; // rsi
  unsigned __int8 v25; // al
  unsigned __int8 **v26; // rax
  unsigned __int8 **v27; // rdx
  unsigned int v28; // ecx
  unsigned __int8 **v29; // rax
  unsigned __int8 *v30; // r8
  bool v31; // al
  bool v32; // dl
  int v33; // edx
  unsigned int v34; // edi
  unsigned __int8 **v35; // rcx
  unsigned __int8 *v36; // r8
  __int64 *v37; // rax
  int v38; // ecx
  int v39; // r9d
  int v40; // eax
  int v41; // r9d
  __int64 *v42; // [rsp+8h] [rbp-38h]

  v7 = sub_BD42C0(*a2, (__int64)a2);
  v8 = sub_98ACB0(v7, 6u);
  v9 = sub_BD42C0(*a3, 6);
  v10 = sub_98ACB0(v9, 6u);
  v11 = *v8;
  if ( *v8 > 3u )
  {
    v16 = 0;
    v12 = (__int64)v10;
    if ( *v10 > 3u )
    {
LABEL_19:
      if ( v11 != 61 )
        goto LABEL_20;
      goto LABEL_37;
    }
    goto LABEL_67;
  }
  v12 = 0;
  if ( *v10 < 4u )
    v12 = (__int64)v10;
  if ( !*(_BYTE *)(a1 + 68) )
  {
    v16 = sub_C8CA60(a1 + 40, (__int64)v8);
    if ( v16 )
      v16 = (__int64 *)v8;
    if ( !v12 )
      goto LABEL_33;
LABEL_67:
    if ( *(_BYTE *)(a1 + 68) )
    {
      v13 = *(__int64 ***)(a1 + 48);
      v14 = &v13[*(unsigned int *)(a1 + 60)];
      if ( v13 == v14 )
        goto LABEL_33;
LABEL_12:
      while ( (__int64 *)v12 != *v13 )
      {
        if ( v14 == ++v13 )
          goto LABEL_33;
      }
    }
    else
    {
      v42 = v16;
      v37 = sub_C8CA60(a1 + 40, v12);
      v16 = v42;
      if ( !v37 )
        goto LABEL_33;
    }
    if ( v16 )
    {
      if ( v16 != (__int64 *)v12 )
        return 0;
      goto LABEL_18;
    }
    if ( byte_4F86C08 )
    {
      if ( v12 )
        return 0;
LABEL_18:
      v11 = *v8;
      goto LABEL_19;
    }
    if ( v12 )
    {
      v16 = (__int64 *)v12;
      v18 = (__int64)v8;
LABEL_17:
      if ( (unsigned __int8)sub_D1C170((__int64 *)a1, (__int64)v16, v18, a5) )
        return 0;
      goto LABEL_18;
    }
LABEL_35:
    v25 = *v8;
    goto LABEL_36;
  }
  v13 = *(__int64 ***)(a1 + 48);
  v14 = &v13[*(unsigned int *)(a1 + 60)];
  if ( v13 != v14 )
  {
    v15 = *(__int64 ***)(a1 + 48);
    while ( 1 )
    {
      v16 = *v15;
      if ( v8 == (unsigned __int8 *)*v15 )
        break;
      if ( v14 == ++v15 )
      {
        v16 = 0;
        break;
      }
    }
    if ( v12 )
      goto LABEL_12;
LABEL_33:
    if ( byte_4F86C08 )
    {
      if ( v16 )
        return 0;
      goto LABEL_35;
    }
    v18 = (__int64)v10;
    if ( v16 )
      goto LABEL_17;
    goto LABEL_35;
  }
  v16 = 0;
  if ( v12 )
    goto LABEL_33;
  v25 = *v8;
LABEL_36:
  if ( v25 != 61 )
    goto LABEL_20;
LABEL_37:
  v20 = (unsigned __int8 *)*((_QWORD *)v8 - 4);
  if ( *v20 == 3 )
  {
    if ( *(_BYTE *)(a1 + 172) )
    {
      v26 = *(unsigned __int8 ***)(a1 + 152);
      v27 = &v26[*(unsigned int *)(a1 + 164)];
      if ( v26 == v27 )
        goto LABEL_20;
      while ( v20 != *v26 )
      {
        if ( v27 == ++v26 )
          goto LABEL_20;
      }
LABEL_61:
      if ( *v10 != 61 )
        goto LABEL_62;
      v19 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
      if ( *v19 != 3 )
        goto LABEL_62;
      goto LABEL_22;
    }
    if ( sub_C8CA60(a1 + 144, *((_QWORD *)v8 - 4)) )
      goto LABEL_61;
  }
LABEL_20:
  if ( *v10 != 61 )
    goto LABEL_45;
  v19 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
  v20 = 0;
  if ( *v19 != 3 )
    goto LABEL_88;
LABEL_22:
  if ( *(_BYTE *)(a1 + 172) )
  {
    v21 = *(unsigned __int8 ***)(a1 + 152);
    v22 = &v21[*(unsigned int *)(a1 + 164)];
    if ( v21 == v22 )
    {
LABEL_87:
      if ( !v20 )
      {
LABEL_88:
        v23 = *(_DWORD *)(a1 + 264);
        if ( v23 )
        {
          v24 = *(_QWORD *)(a1 + 248);
          v19 = 0;
          goto LABEL_47;
        }
LABEL_53:
        v31 = 0;
        v20 = 0;
        goto LABEL_54;
      }
LABEL_62:
      v24 = *(_QWORD *)(a1 + 248);
      v23 = *(_DWORD *)(a1 + 264);
      goto LABEL_63;
    }
    do
    {
      if ( *v21 == v19 )
      {
        if ( v20 )
          goto LABEL_49;
        goto LABEL_28;
      }
      ++v21;
    }
    while ( v22 != v21 );
    if ( v20 )
      goto LABEL_62;
LABEL_45:
    v23 = *(_DWORD *)(a1 + 264);
    v24 = *(_QWORD *)(a1 + 248);
    if ( v23 )
    {
      v19 = 0;
      goto LABEL_47;
    }
    goto LABEL_53;
  }
  if ( !sub_C8CA60(a1 + 144, (__int64)v19) )
    goto LABEL_87;
  if ( v20 )
    return v19 == v20;
LABEL_28:
  v23 = *(_DWORD *)(a1 + 264);
  v24 = *(_QWORD *)(a1 + 248);
  if ( !v23 )
    return !byte_4F86C08;
LABEL_47:
  v28 = (v23 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v29 = (unsigned __int8 **)(v24 + 16LL * v28);
  v30 = *v29;
  if ( v8 != *v29 )
  {
    v40 = 1;
    while ( v30 != (unsigned __int8 *)-4096LL )
    {
      v41 = v40 + 1;
      v28 = (v23 - 1) & (v40 + v28);
      v29 = (unsigned __int8 **)(v24 + 16LL * v28);
      v30 = *v29;
      if ( v8 == *v29 )
        goto LABEL_48;
      v40 = v41;
    }
    if ( v19 )
      return !byte_4F86C08;
    v20 = 0;
LABEL_63:
    v31 = v20 != 0;
    if ( v23 )
    {
      v33 = v23 - 1;
      v34 = v33 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v35 = (unsigned __int8 **)(v24 + 16LL * v34);
      v36 = *v35;
      if ( v10 == *v35 )
      {
LABEL_65:
        v19 = v35[1];
        goto LABEL_49;
      }
      v38 = 1;
      while ( v36 != (unsigned __int8 *)-4096LL )
      {
        v39 = v38 + 1;
        v34 = v33 & (v38 + v34);
        v35 = (unsigned __int8 **)(v24 + 16LL * v34);
        v36 = *v35;
        if ( v10 == *v35 )
          goto LABEL_65;
        v38 = v39;
      }
    }
LABEL_54:
    v19 = 0;
    v32 = 0;
    return !byte_4F86C08 || !v31 && !v32 || v20 == v19;
  }
LABEL_48:
  v20 = v29[1];
  if ( !v19 )
    goto LABEL_63;
LABEL_49:
  v31 = v20 != 0;
  v32 = v19 != 0;
  if ( !v20 || !v19 )
    return !byte_4F86C08 || !v31 && !v32 || v20 == v19;
  return v19 == v20;
}
