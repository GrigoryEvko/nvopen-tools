// Function: sub_244A810
// Address: 0x244a810
//
void **__fastcall sub_244A810(void **p_src, _BYTE *a2)
{
  _BYTE *v2; // rcx
  __int64 v3; // r15
  _BYTE *v4; // rbx
  __int64 v5; // r13
  _BYTE *v6; // rax
  unsigned __int8 *v7; // rax
  _BYTE *v8; // rax
  _BYTE *v9; // r12
  _BYTE *v10; // rbx
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  unsigned __int64 v13; // r14
  char *v14; // rdi
  char *v15; // rbx
  size_t v16; // rdx
  char *v17; // rcx
  void **v19; // [rsp+8h] [rbp-98h]
  _BYTE *v20; // [rsp+10h] [rbp-90h]
  _BYTE *v21; // [rsp+18h] [rbp-88h]
  char *v22; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v23; // [rsp+28h] [rbp-78h] BYREF
  void *src; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v25; // [rsp+38h] [rbp-68h]
  _BYTE *v26; // [rsp+40h] [rbp-60h]
  unsigned __int64 v27; // [rsp+48h] [rbp-58h] BYREF
  _BYTE *v28; // [rsp+50h] [rbp-50h]
  _BYTE *v29; // [rsp+58h] [rbp-48h]
  int v30; // [rsp+60h] [rbp-40h]

  v2 = (_BYTE *)*((_QWORD *)a2 + 10);
  v19 = p_src;
  src = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v20 = a2 + 72;
  v21 = v2;
  if ( a2 + 72 == v2 )
  {
    *p_src = 0;
    p_src[2] = 0;
    p_src[1] = 0;
    return v19;
  }
  do
  {
    v3 = *((_QWORD *)v21 + 4);
    v4 = v21 + 24;
    v21 = (_BYTE *)*((_QWORD *)v21 + 1);
    while ( v4 != (_BYTE *)v3 )
    {
LABEL_4:
      v5 = v3;
      v3 = *(_QWORD *)(v3 + 8);
      switch ( *(_BYTE *)(v5 - 24) )
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
          p_src = (void **)(v5 - 24);
          if ( !sub_B491E0(v5 - 24) )
            continue;
          v23 = (unsigned __int8 *)(v5 - 24);
          a2 = v25;
          if ( v25 == v26 )
          {
            p_src = &src;
            sub_2445670((__int64)&src, v25, &v23);
          }
          else
          {
            if ( v25 )
            {
              *(_QWORD *)v25 = v5 - 24;
              a2 = v25;
            }
            a2 += 8;
            v25 = a2;
          }
          if ( v30 != 1 )
            continue;
          p_src = (void **)(v5 - 24);
          if ( !sub_B491E0(v5 - 24) )
            continue;
          v6 = *(_BYTE **)(v5 - 56);
          if ( *v6 != 61 )
            continue;
          p_src = (void **)*((_QWORD *)v6 - 4);
          v7 = sub_BD4070((unsigned __int8 *)p_src, (__int64)a2);
          if ( !v7 || *v7 <= 0x1Cu )
            continue;
          v23 = v7;
          a2 = v28;
          if ( v28 != v29 )
          {
            if ( v28 )
            {
              *(_QWORD *)v28 = v7;
              a2 = v28;
            }
            a2 += 8;
            v28 = a2;
            if ( v4 == (_BYTE *)v3 )
              goto LABEL_19;
            goto LABEL_4;
          }
          p_src = (void **)&v27;
          sub_24454E0((__int64)&v27, v28, &v23);
          break;
        case 0x55:
          a2 = (_BYTE *)(v5 - 24);
          p_src = &src;
          sub_2449C10((__int64)&src, v5 - 24);
          continue;
        default:
          BUG();
      }
    }
LABEL_19:
    ;
  }
  while ( v20 != v21 );
  v8 = v25;
  v9 = src;
  v10 = v25;
  *v19 = 0;
  v19[1] = 0;
  v19[2] = 0;
  v11 = v10 - v9;
  if ( v11 )
  {
    if ( v11 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(p_src, a2, v19);
    v12 = sub_22077B0(v11);
    v9 = src;
    v13 = v27;
    v14 = (char *)v12;
    v8 = v25;
    v15 = &v14[v11];
    v16 = v25 - (_BYTE *)src;
    v17 = &v14[v25 - (_BYTE *)src];
  }
  else
  {
    v13 = v27;
    v15 = 0;
    v17 = 0;
    v16 = 0;
    v14 = 0;
  }
  *v19 = v14;
  v19[1] = v14;
  v19[2] = v15;
  if ( v8 != v9 )
  {
    v22 = v17;
    memmove(v14, v9, v16);
    v19[1] = v22;
    if ( v13 )
      goto LABEL_25;
    goto LABEL_29;
  }
  v19[1] = v17;
  if ( v13 )
  {
LABEL_25:
    j_j___libc_free_0(v13);
    v9 = src;
  }
  if ( v9 )
LABEL_29:
    j_j___libc_free_0((unsigned __int64)v9);
  return v19;
}
