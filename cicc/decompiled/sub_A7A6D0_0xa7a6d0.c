// Function: sub_A7A6D0
// Address: 0xa7a6d0
//
__int64 __fastcall sub_A7A6D0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rax
  int v7; // eax
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r13
  int v21; // r13d
  int v22; // r13d
  unsigned int v23; // r13d
  int v24; // eax
  unsigned __int16 v25; // ax
  unsigned __int8 v26; // dl
  unsigned __int8 v27; // r13
  unsigned __int16 v28; // ax
  char v29; // dl
  unsigned int v30; // eax
  unsigned __int16 v31; // [rsp+Eh] [rbp-152h]
  unsigned __int64 v32; // [rsp+10h] [rbp-150h]
  __int64 *v33; // [rsp+28h] [rbp-138h]
  __int64 v34; // [rsp+30h] [rbp-130h]
  __int64 *v35; // [rsp+38h] [rbp-128h]
  __int64 v36; // [rsp+58h] [rbp-108h] BYREF
  __int64 v37; // [rsp+60h] [rbp-100h] BYREF
  __int64 v38; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v39; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v40; // [rsp+78h] [rbp-E8h]
  __int64 v41; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v42; // [rsp+88h] [rbp-D8h]
  __m128i v43; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v45; // [rsp+A8h] [rbp-B8h]
  __m128i v46; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+C0h] [rbp-A0h]
  unsigned int v48; // [rsp+C8h] [rbp-98h]
  __m128i v49; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v50; // [rsp+E0h] [rbp-80h]
  _BYTE v51[120]; // [rsp+E8h] [rbp-78h] BYREF

  v3 = *a1;
  v33 = (__int64 *)a2;
  v36 = a3;
  if ( v3 == a3 )
  {
    v49.m128i_i64[0] = v3;
    v49.m128i_i8[8] = 1;
    v46 = _mm_loadu_si128(&v49);
    return v46.m128i_i64[0];
  }
  v49.m128i_i64[0] = a2;
  v49.m128i_i64[1] = (__int64)v51;
  v50 = 0x800000000LL;
  v4 = (__int64 *)sub_A73280(a1);
  v34 = sub_A73290(a1);
  v5 = (__int64 *)sub_A73280(&v36);
  v6 = sub_A73290(&v36);
  v35 = (__int64 *)v6;
  if ( v4 == (__int64 *)v34 && v5 == (__int64 *)v6 )
  {
LABEL_27:
    a2 = (__int64)&v49;
    v13 = sub_A7A280(v33, (__int64)&v49);
    v43.m128i_i8[8] = 1;
    v43.m128i_i64[0] = v13;
    v46 = _mm_loadu_si128(&v43);
    goto LABEL_19;
  }
  while ( 1 )
  {
    v37 = 0;
    v38 = 0;
    if ( v35 == v5 )
      break;
    if ( (__int64 *)v34 == v4 )
      goto LABEL_14;
    a2 = *v5;
    v7 = sub_A730B0(v4, *v5);
    if ( v7 )
    {
      if ( v7 >= 0 )
      {
LABEL_14:
        v10 = *v5++;
        v37 = v10;
        goto LABEL_15;
      }
      break;
    }
    v8 = *v4;
    ++v5;
    ++v4;
    v37 = v8;
    v38 = *(v5 - 1);
    if ( sub_A71840((__int64)&v37) )
      goto LABEL_16;
LABEL_8:
    v9 = sub_A71AE0(&v37);
    if ( v38 )
    {
      if ( sub_A71A70(v9) )
      {
        a2 = v9;
        sub_A77B20((__int64 **)&v49, v9);
      }
      else if ( sub_A71A90(v9) )
      {
        v32 = sub_A71B80(&v38);
        v14 = sub_A71B80(&v37);
        v15 = v32;
        a2 = v9;
        if ( v32 > v14 )
          v15 = v14;
        sub_A77B60((__int64 **)&v49, v9, v15);
      }
      else if ( sub_A71AB0(v9) )
      {
        switch ( v9 )
        {
          case 'V':
            v25 = sub_A71F30(&v38);
            v26 = 0;
            if ( HIBYTE(v25) )
              v26 = v25;
            v27 = v26;
            v28 = sub_A71F30(&v37);
            v29 = 0;
            if ( HIBYTE(v28) )
            {
              v29 = v28;
              if ( v27 < (unsigned __int8)v28 )
                v29 = v27;
            }
            v30 = v31;
            LOBYTE(v30) = v29;
            BYTE1(v30) = 1;
            a2 = v30;
            v31 = v30;
            sub_A77B90((__int64 **)&v49, v30);
            break;
          case 'Y':
            v23 = sub_A71E10(&v38);
            v24 = sub_A71E10(&v37);
            a2 = v24 | v23;
            sub_A77CE0((__int64 **)&v49, v24 | v23);
            break;
          case '\\':
            v22 = sub_A71E40(&v38);
            a2 = (unsigned int)sub_A71E40(&v37) | v22;
            sub_A77CD0((__int64 **)&v49, a2);
            break;
          case ']':
            v21 = sub_A71E30(&v38);
            a2 = (unsigned int)sub_A71E30(&v37) & v21;
            sub_A77D00((__int64 **)&v49, a2);
            break;
          case 'a':
            v17 = sub_A72AA0(&v37);
            v18 = v17;
            v40 = *(_DWORD *)(v17 + 8);
            if ( v40 > 0x40 )
              sub_C43780(&v39, v17);
            else
              v39 = *(_QWORD *)v17;
            v42 = *(_DWORD *)(v18 + 24);
            if ( v42 > 0x40 )
              sub_C43780(&v41, v18 + 16);
            else
              v41 = *(_QWORD *)(v18 + 16);
            v19 = sub_A72AA0(&v38);
            v20 = v19;
            v43.m128i_i32[2] = *(_DWORD *)(v19 + 8);
            if ( v43.m128i_i32[2] > 0x40u )
              sub_C43780(&v43, v19);
            else
              v43.m128i_i64[0] = *(_QWORD *)v19;
            v45 = *(_DWORD *)(v20 + 24);
            if ( v45 > 0x40 )
              sub_C43780(&v44, v20 + 16);
            else
              v44 = *(_QWORD *)(v20 + 16);
            a2 = (__int64)&v39;
            sub_AB3510(&v46, &v39, &v43, 0);
            if ( !(unsigned __int8)sub_AAF760(&v46) )
            {
              a2 = (__int64)&v46;
              sub_A78C10(&v49, (__int64)&v46);
            }
            if ( v48 > 0x40 && v47 )
              j_j___libc_free_0_0(v47);
            if ( v46.m128i_i32[2] > 0x40u && v46.m128i_i64[0] )
              j_j___libc_free_0_0(v46.m128i_i64[0]);
            if ( v45 > 0x40 && v44 )
              j_j___libc_free_0_0(v44);
            if ( v43.m128i_i32[2] > 0x40u && v43.m128i_i64[0] )
              j_j___libc_free_0_0(v43.m128i_i64[0]);
            if ( v42 > 0x40 && v41 )
              j_j___libc_free_0_0(v41);
            if ( v40 > 0x40 && v39 )
              j_j___libc_free_0_0(v39);
            break;
          default:
            BUG();
        }
      }
      else
      {
        a2 = v38;
        if ( !v38 )
          goto LABEL_18;
        if ( v38 != v37 )
          goto LABEL_18;
        sub_A77670((__int64)&v49, v38);
        if ( v9 == 81 )
        {
          a2 = 86;
          v16 = sub_A734C0(&v36, 86);
          if ( v16 != sub_A734C0(a1, 86) )
            goto LABEL_18;
        }
      }
    }
    else if ( sub_A71A50(v9) )
    {
      goto LABEL_18;
    }
LABEL_25:
    if ( v4 == (__int64 *)v34 && v5 == v35 )
      goto LABEL_27;
  }
  v12 = *v4++;
  v37 = v12;
LABEL_15:
  if ( !sub_A71840((__int64)&v37) )
    goto LABEL_8;
LABEL_16:
  a2 = v38;
  if ( v38 && v38 == v37 )
  {
    sub_A77670((__int64)&v49, v38);
    goto LABEL_25;
  }
LABEL_18:
  v46.m128i_i8[8] = 0;
LABEL_19:
  if ( (_BYTE *)v49.m128i_i64[1] != v51 )
    _libc_free(v49.m128i_i64[1], a2);
  return v46.m128i_i64[0];
}
