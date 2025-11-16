// Function: sub_24DB2C0
// Address: 0x24db2c0
//
__int64 __fastcall sub_24DB2C0(void **p_src, _BYTE *a2, int a3)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r13
  bool v10; // al
  __int64 v11; // r9
  _BYTE *v12; // rax
  unsigned __int8 *v13; // rax
  __int64 v14; // rdx
  void *v15; // r15
  char *v16; // rax
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // r13
  __int64 *v19; // rbx
  __int64 *v20; // r15
  const __m128i *i; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r14
  __int64 v30; // rsi
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // r13
  bool v36; // al
  __int64 v37; // r9
  _BYTE *v38; // rax
  unsigned __int8 *v39; // rax
  _BYTE *v40; // rax
  _BYTE *v41; // r13
  __int64 v42; // rax
  __int64 v43; // rcx
  size_t v44; // rdx
  __int64 *v45; // r15
  __int64 v46; // r13
  __int64 *j; // r14
  __int64 v48; // rbx
  unsigned __int8 *v49; // rdi
  int v50; // eax
  unsigned __int64 v51; // rax
  __int64 v52; // [rsp+8h] [rbp-C8h]
  __int64 v53; // [rsp+8h] [rbp-C8h]
  __int64 v54; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v55; // [rsp+10h] [rbp-C0h]
  __int64 v56; // [rsp+10h] [rbp-C0h]
  __int64 v57; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v58; // [rsp+10h] [rbp-C0h]
  __int64 *dest; // [rsp+18h] [rbp-B8h]
  const __m128i *destb; // [rsp+18h] [rbp-B8h]
  __int64 *desta; // [rsp+18h] [rbp-B8h]
  void *src; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE *v63; // [rsp+28h] [rbp-A8h]
  _BYTE *v64; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v65; // [rsp+38h] [rbp-98h] BYREF
  _BYTE *v66; // [rsp+40h] [rbp-90h]
  _BYTE *v67; // [rsp+48h] [rbp-88h]
  int v68; // [rsp+50h] [rbp-80h]
  __m128i v69; // [rsp+60h] [rbp-70h] BYREF
  __int64 v70; // [rsp+70h] [rbp-60h]
  void *v71; // [rsp+78h] [rbp-58h] BYREF
  _BYTE *v72; // [rsp+80h] [rbp-50h]
  _BYTE *v73; // [rsp+88h] [rbp-48h]
  int v74; // [rsp+90h] [rbp-40h]

  v3 = (__int64)p_src;
  *p_src = 0;
  p_src[1] = 0;
  p_src[2] = 0;
  v4 = *(_QWORD *)a2;
  dest = *(__int64 **)a2;
  if ( a3 == 1 )
  {
    *(_QWORD *)(v4 + 32) = p_src;
    v24 = v4 + 16;
    v25 = *(_QWORD *)(v4 + 16);
    v26 = *(_QWORD *)(v25 + 80);
    v56 = v25 + 72;
    if ( v25 + 72 == v26 )
    {
LABEL_45:
      dest[4] = 0;
      return v3;
    }
LABEL_37:
    v27 = v26;
    v26 = *(_QWORD *)(v26 + 8);
    v28 = *(_QWORD *)(v27 + 32);
    v29 = v27 + 24;
    while ( 1 )
    {
LABEL_39:
      if ( v29 == v28 )
      {
LABEL_44:
        if ( v56 == v26 )
          goto LABEL_45;
        goto LABEL_37;
      }
      while ( 1 )
      {
        v30 = v28;
        v28 = *(_QWORD *)(v28 + 8);
        v31 = *(unsigned __int8 *)(v30 - 24);
        if ( v31 == 85 )
          break;
        if ( (unsigned int)(v31 - 29) <= 0x38 )
        {
          if ( (unsigned int)(v31 - 30) > 0x36 )
LABEL_98:
            BUG();
          goto LABEL_39;
        }
        if ( (unsigned int)(v31 - 86) > 0xA )
          goto LABEL_98;
        if ( v29 == v28 )
          goto LABEL_44;
      }
      sub_24DAF10(v24, v30 - 24);
    }
  }
  if ( !a3 )
  {
    v6 = dest[1];
    v68 = 0;
    src = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v52 = v6 + 72;
    v54 = *(_QWORD *)(v6 + 80);
    if ( v6 + 72 == v54 )
    {
      v19 = 0;
      v18 = 0;
      goto LABEL_30;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(v54 + 32);
      v8 = v54 + 24;
      v54 = *(_QWORD *)(v54 + 8);
      while ( v8 != v7 )
      {
LABEL_8:
        v9 = v7;
        v7 = *(_QWORD *)(v7 + 8);
        switch ( *(_BYTE *)(v9 - 24) )
        {
          case 0x1E:
          case 0x1F:
          case 0x20:
          case 0x21:
          case 0x23:
          case 0x24:
          case 0x25:
          case 0x26:
          case 0x27:
          case 0x29:
          case 0x2A:
          case 0x2B:
          case 0x2C:
          case 0x2D:
          case 0x2E:
          case 0x2F:
          case 0x30:
          case 0x31:
          case 0x32:
          case 0x33:
          case 0x34:
          case 0x35:
          case 0x36:
          case 0x37:
          case 0x38:
          case 0x39:
          case 0x3A:
          case 0x3B:
          case 0x3C:
          case 0x3D:
          case 0x3E:
          case 0x3F:
          case 0x40:
          case 0x41:
          case 0x42:
          case 0x43:
          case 0x44:
          case 0x45:
          case 0x46:
          case 0x47:
          case 0x48:
          case 0x49:
          case 0x4A:
          case 0x4B:
          case 0x4C:
          case 0x4D:
          case 0x4E:
          case 0x4F:
          case 0x50:
          case 0x51:
          case 0x52:
          case 0x53:
          case 0x54:
          case 0x56:
          case 0x57:
          case 0x58:
          case 0x59:
          case 0x5A:
          case 0x5B:
          case 0x5C:
          case 0x5D:
          case 0x5E:
          case 0x5F:
          case 0x60:
            continue;
          case 0x22:
          case 0x28:
            p_src = (void **)(v9 - 24);
            v10 = sub_B491E0(v9 - 24);
            v11 = v9 - 24;
            if ( !v10 )
              continue;
            v69.m128i_i64[0] = v9 - 24;
            a2 = v63;
            if ( v63 == v64 )
            {
              p_src = &src;
              sub_2445670((__int64)&src, v63, &v69);
              v11 = v9 - 24;
            }
            else
            {
              if ( v63 )
              {
                *(_QWORD *)v63 = v9 - 24;
                a2 = v63;
              }
              a2 += 8;
              v63 = a2;
            }
            if ( v68 != 1 )
              continue;
            p_src = (void **)v11;
            if ( !sub_B491E0(v11) )
              continue;
            v12 = *(_BYTE **)(v9 - 56);
            if ( *v12 != 61 )
              continue;
            p_src = (void **)*((_QWORD *)v12 - 4);
            v13 = sub_BD4070((unsigned __int8 *)p_src, (__int64)a2);
            if ( !v13 || *v13 <= 0x1Cu )
              continue;
            v69.m128i_i64[0] = (__int64)v13;
            a2 = v66;
            if ( v66 != v67 )
            {
              if ( v66 )
              {
                *(_QWORD *)v66 = v13;
                a2 = v66;
              }
              a2 += 8;
              v66 = a2;
              if ( v8 == v7 )
                goto LABEL_23;
              goto LABEL_8;
            }
            p_src = (void **)&v65;
            sub_24454E0((__int64)&v65, v66, &v69);
            break;
          case 0x55:
            a2 = (_BYTE *)(v9 - 24);
            p_src = &src;
            sub_2449C10((__int64)&src, v9 - 24);
            continue;
          default:
            goto LABEL_98;
        }
      }
LABEL_23:
      v14 = v54;
      if ( v52 == v54 )
      {
        v15 = src;
        v55 = v63 - (_BYTE *)src;
        if ( v63 == src )
        {
          v17 = v65;
          v19 = 0;
          v18 = 0;
        }
        else
        {
          if ( v55 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_97;
          v16 = (char *)sub_22077B0(v55);
          v15 = src;
          v17 = v65;
          v18 = (unsigned __int64)v16;
          v19 = (__int64 *)&v16[v63 - (_BYTE *)src];
          if ( v63 != src )
          {
            memmove(v16, src, v63 - (_BYTE *)src);
            if ( v17 )
              goto LABEL_28;
            goto LABEL_90;
          }
        }
        if ( v17 )
        {
LABEL_28:
          j_j___libc_free_0(v17);
          v15 = src;
        }
        if ( v15 )
LABEL_90:
          j_j___libc_free_0((unsigned __int64)v15);
LABEL_30:
        v20 = (__int64 *)v18;
        for ( i = &v69; v19 != v20; i = destb )
        {
          v22 = *v20++;
          destb = i;
          v23 = *(_QWORD *)(v22 - 32);
          v69.m128i_i64[1] = v22;
          v70 = v22;
          v69.m128i_i64[0] = v23;
          sub_24DAD90(v3, i);
        }
        if ( v18 )
          j_j___libc_free_0(v18);
        return v3;
      }
    }
  }
  if ( a3 != 2 )
    return v3;
  v32 = *dest;
  v69 = 0u;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 1;
  v53 = v32 + 72;
  v57 = *(_QWORD *)(v32 + 80);
  if ( v32 + 72 == v57 )
    return v3;
  do
  {
    v33 = *(_QWORD *)(v57 + 32);
    v34 = v57 + 24;
    v57 = *(_QWORD *)(v57 + 8);
    while ( v34 != v33 )
    {
LABEL_50:
      v35 = v33;
      v33 = *(_QWORD *)(v33 + 8);
      switch ( *(_BYTE *)(v35 - 24) )
      {
        case 0x1E:
        case 0x1F:
        case 0x20:
        case 0x21:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0x27:
        case 0x29:
        case 0x2A:
        case 0x2B:
        case 0x2C:
        case 0x2D:
        case 0x2E:
        case 0x2F:
        case 0x30:
        case 0x31:
        case 0x32:
        case 0x33:
        case 0x34:
        case 0x35:
        case 0x36:
        case 0x37:
        case 0x38:
        case 0x39:
        case 0x3A:
        case 0x3B:
        case 0x3C:
        case 0x3D:
        case 0x3E:
        case 0x3F:
        case 0x40:
        case 0x41:
        case 0x42:
        case 0x43:
        case 0x44:
        case 0x45:
        case 0x46:
        case 0x47:
        case 0x48:
        case 0x49:
        case 0x4A:
        case 0x4B:
        case 0x4C:
        case 0x4D:
        case 0x4E:
        case 0x4F:
        case 0x50:
        case 0x51:
        case 0x52:
        case 0x53:
        case 0x54:
        case 0x56:
        case 0x57:
        case 0x58:
        case 0x59:
        case 0x5A:
        case 0x5B:
        case 0x5C:
        case 0x5D:
        case 0x5E:
        case 0x5F:
        case 0x60:
          continue;
        case 0x22:
        case 0x28:
          p_src = (void **)(v35 - 24);
          v36 = sub_B491E0(v35 - 24);
          v37 = v35 - 24;
          if ( !v36 )
            continue;
          src = (void *)(v35 - 24);
          a2 = (_BYTE *)v69.m128i_i64[1];
          if ( v69.m128i_i64[1] == v70 )
          {
            p_src = (void **)&v69;
            sub_2445670((__int64)&v69, (_BYTE *)v69.m128i_i64[1], &src);
            v37 = v35 - 24;
          }
          else
          {
            if ( v69.m128i_i64[1] )
            {
              *(_QWORD *)v69.m128i_i64[1] = v35 - 24;
              a2 = (_BYTE *)v69.m128i_i64[1];
            }
            a2 += 8;
            v69.m128i_i64[1] = (__int64)a2;
          }
          if ( v74 != 1 )
            continue;
          p_src = (void **)v37;
          if ( !sub_B491E0(v37) )
            continue;
          v38 = *(_BYTE **)(v35 - 56);
          if ( *v38 != 61 )
            continue;
          p_src = (void **)*((_QWORD *)v38 - 4);
          v39 = sub_BD4070((unsigned __int8 *)p_src, (__int64)a2);
          if ( !v39 || *v39 <= 0x1Cu )
            continue;
          src = v39;
          a2 = v72;
          if ( v72 != v73 )
          {
            if ( v72 )
            {
              *(_QWORD *)v72 = v39;
              a2 = v72;
            }
            a2 += 8;
            v72 = a2;
            if ( v34 == v33 )
              goto LABEL_65;
            goto LABEL_50;
          }
          p_src = &v71;
          sub_24454E0((__int64)&v71, v72, &src);
          break;
        case 0x55:
          a2 = (_BYTE *)(v35 - 24);
          p_src = (void **)&v69;
          sub_2449C10((__int64)&v69, v35 - 24);
          continue;
        default:
          goto LABEL_98;
      }
    }
LABEL_65:
    v14 = v57;
  }
  while ( v53 != v57 );
  v40 = v72;
  v41 = v71;
  v58 = v72 - (_BYTE *)v71;
  if ( v72 != v71 )
  {
    if ( v58 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v42 = sub_22077B0(v58);
      v41 = v71;
      v43 = v42;
      desta = (__int64 *)v42;
      v40 = v72;
      v44 = v72 - (_BYTE *)v71;
      v45 = (__int64 *)(v43 + v72 - (_BYTE *)v71);
      goto LABEL_69;
    }
LABEL_97:
    sub_4261EA(p_src, a2, v14);
  }
  v45 = 0;
  v44 = 0;
  desta = 0;
LABEL_69:
  if ( v41 != v40 )
  {
    memmove(desta, v41, v44);
    goto LABEL_71;
  }
  if ( v41 )
LABEL_71:
    j_j___libc_free_0((unsigned __int64)v41);
  if ( v69.m128i_i64[0] )
    j_j___libc_free_0(v69.m128i_u64[0]);
  v46 = 0x100060000000001LL;
  for ( j = desta; v45 != j; ++j )
  {
    v48 = *j;
    v49 = (unsigned __int8 *)sub_B46B10(*j, 0);
    if ( v49 )
    {
      while ( 1 )
      {
        v50 = *v49;
        if ( (_BYTE)v50 != 84 )
        {
          v51 = (unsigned int)(v50 - 39);
          if ( (unsigned int)v51 > 0x38 || !_bittest64(&v46, v51) )
            break;
        }
        v49 = (unsigned __int8 *)sub_B46B10((__int64)v49, 0);
        if ( !v49 )
          goto LABEL_82;
      }
      v69.m128i_i64[1] = (__int64)v49;
      v69.m128i_i64[0] = v48;
      v70 = v48;
      sub_24DAD90(v3, &v69);
    }
LABEL_82:
    ;
  }
  if ( desta )
    j_j___libc_free_0((unsigned __int64)desta);
  return v3;
}
