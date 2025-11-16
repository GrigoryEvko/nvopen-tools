// Function: sub_13F3D10
// Address: 0x13f3d10
//
__int64 __fastcall sub_13F3D10(_QWORD *a1, __int64 a2, unsigned __int8 a3, __int64 a4)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  __int64 v6; // r12
  unsigned __int8 v7; // bl
  _QWORD *v8; // rax
  char v9; // dl
  __int64 v10; // r8
  __int64 v11; // r14
  unsigned __int8 v12; // al
  __int64 v13; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  _QWORD *v19; // rsi
  unsigned int v20; // edi
  _QWORD *v21; // rcx
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 *v24; // rax
  __int64 *v25; // rdx
  unsigned __int64 v26; // rdi
  __int64 v27; // rax
  char v28; // dl
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 *v31; // rsi
  __int64 *v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // [rsp-10h] [rbp-B0h]
  __int64 v37; // [rsp+18h] [rbp-88h] BYREF
  __m128i v38; // [rsp+20h] [rbp-80h] BYREF
  __int64 *v39; // [rsp+30h] [rbp-70h]
  __int64 v40; // [rsp+38h] [rbp-68h]
  __int64 v41; // [rsp+40h] [rbp-60h]
  _BYTE v42[88]; // [rsp+48h] [rbp-58h] BYREF

  while ( 1 )
  {
    v4 = (_QWORD *)a2;
    v5 = a1;
    v6 = a4;
    v7 = a3;
    v8 = *(_QWORD **)(a4 + 8);
    if ( *(_QWORD **)(a4 + 16) != v8 )
      goto LABEL_2;
    v19 = &v8[*(unsigned int *)(a4 + 28)];
    v20 = *(_DWORD *)(a4 + 28);
    if ( v8 == v19 )
      goto LABEL_34;
    v21 = 0;
    do
    {
      if ( v4 == (_QWORD *)*v8 )
        return sub_1599EF0(*v4);
      if ( *v8 == -2 )
        v21 = v8;
      ++v8;
    }
    while ( v19 != v8 );
    if ( !v21 )
    {
LABEL_34:
      if ( v20 >= *(_DWORD *)(v6 + 24) )
      {
LABEL_2:
        sub_16CCBA0(v6, v4);
        if ( !v9 )
          return sub_1599EF0(*v4);
      }
      else
      {
        *(_DWORD *)(v6 + 28) = v20 + 1;
        *v19 = v4;
        ++*(_QWORD *)v6;
      }
      if ( !v7 )
        goto LABEL_4;
      goto LABEL_11;
    }
    *v21 = v4;
    --*(_DWORD *)(v6 + 32);
    ++*(_QWORD *)v6;
    if ( !a3 )
    {
LABEL_4:
      v11 = sub_1649C60(v4);
      v12 = *(_BYTE *)(v11 + 16);
      if ( v12 <= 0x17u )
        goto LABEL_5;
      goto LABEL_12;
    }
LABEL_11:
    v11 = sub_14AD280(v4, v5[21], 6);
    v12 = *(_BYTE *)(v11 + 16);
    if ( v12 <= 0x17u )
    {
LABEL_5:
      if ( v12 != 5 )
        goto LABEL_9;
      v13 = *(unsigned __int16 *)(v11 + 18);
      if ( (unsigned int)(v13 - 36) > 0xC )
      {
        if ( (_DWORD)v13 != 62 )
          goto LABEL_37;
        v33 = sub_1594710(v11);
        v35 = sub_14AC030(*(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)), v33, v34, 0);
        a2 = v35;
        if ( v11 == v35 || !v35 )
          goto LABEL_8;
        goto LABEL_17;
      }
      if ( !(unsigned __int8)sub_15FB8A0(
                               v13,
                               **(_QWORD **)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)),
                               *(_QWORD *)v11,
                               v5[21]) )
        goto LABEL_8;
      a3 = v7;
      a4 = v6;
      a2 = *(_QWORD *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
      goto LABEL_18;
    }
LABEL_12:
    if ( v12 == 54 )
      break;
    if ( v12 == 77 )
    {
      v22 = sub_15F5600(v11);
      if ( v11 == v22 )
        goto LABEL_8;
      goto LABEL_32;
    }
    if ( (unsigned int)v12 - 60 > 0xC )
    {
      if ( v12 != 86 )
        goto LABEL_16;
      v22 = sub_14AC030(*(_QWORD *)(v11 - 24), *(_QWORD *)(v11 + 56), *(unsigned int *)(v11 + 64), 0);
      if ( v11 == v22 )
        goto LABEL_8;
LABEL_32:
      if ( !v22 )
        goto LABEL_8;
      goto LABEL_33;
    }
    if ( !(unsigned __int8)sub_15FB940(v11, v5[21]) )
      goto LABEL_8;
    a2 = *(_QWORD *)(v11 - 24);
    a3 = v7;
    a4 = v6;
LABEL_18:
    a1 = v5;
  }
  v23 = *(_QWORD *)(v11 + 40);
  v38.m128i_i64[0] = 0;
  v37 = v11 + 24;
  v24 = (__int64 *)v42;
  v38.m128i_i64[1] = (__int64)v42;
  v25 = (__int64 *)v42;
  v39 = (__int64 *)v42;
  v40 = 4;
  LODWORD(v41) = 0;
  while ( 1 )
  {
    if ( v24 != v25 )
      goto LABEL_41;
    v31 = &v24[HIDWORD(v40)];
    if ( v24 != v31 )
      break;
LABEL_56:
    if ( HIDWORD(v40) < (unsigned int)v40 )
    {
      ++HIDWORD(v40);
      *v31 = v23;
      ++v38.m128i_i64[0];
      goto LABEL_42;
    }
LABEL_41:
    sub_16CCBA0(&v38, v23);
    v26 = (unsigned __int64)v39;
    v27 = v38.m128i_i64[1];
    if ( !v28 )
      goto LABEL_61;
LABEL_42:
    v29 = sub_13F95E0(v11, v23, (unsigned int)&v37, qword_4F99140[20], v5[22], 0, 0);
    if ( v29 )
    {
      v11 = sub_13F3D10(v5, v29, v7, v6);
      if ( (__int64 *)v38.m128i_i64[1] != v39 )
        _libc_free((unsigned __int64)v39);
      return v11;
    }
    if ( v37 != *(_QWORD *)(v23 + 48) || (v30 = sub_157F120(v23, v23, v36), (v23 = v30) == 0) )
    {
      v27 = v38.m128i_i64[1];
      v26 = (unsigned __int64)v39;
LABEL_61:
      if ( v26 != v27 )
        _libc_free(v26);
      goto LABEL_8;
    }
    v25 = v39;
    v37 = v30 + 40;
    v24 = (__int64 *)v38.m128i_i64[1];
  }
  v32 = 0;
  while ( v23 != *v24 )
  {
    if ( *v24 == -2 )
      v32 = v24;
    if ( v31 == ++v24 )
    {
      if ( !v32 )
        goto LABEL_56;
      *v32 = v23;
      LODWORD(v41) = v41 - 1;
      ++v38.m128i_i64[0];
      goto LABEL_42;
    }
  }
LABEL_8:
  v12 = *(_BYTE *)(v11 + 16);
  if ( v12 > 0x17u )
  {
LABEL_16:
    v15 = v5[21];
    v16 = v5[25];
    v41 = 0;
    v17 = v5[24];
    v18 = v5[23];
    v38.m128i_i64[0] = v15;
    v38.m128i_i64[1] = v16;
    v39 = (__int64 *)v17;
    v40 = v18;
    a2 = sub_13E3350(v11, &v38, 0, 1, v10);
    if ( !a2 )
      return v11;
LABEL_17:
    a3 = v7;
    a4 = v6;
    goto LABEL_18;
  }
LABEL_9:
  if ( v12 > 0x10u )
    return v11;
LABEL_37:
  v22 = sub_14DBA30(v11, v5[21], v5[25]);
  if ( v11 != v22 && v22 )
  {
LABEL_33:
    a3 = v7;
    a4 = v6;
    a2 = v22;
    goto LABEL_18;
  }
  return v11;
}
