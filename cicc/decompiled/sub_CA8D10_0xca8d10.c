// Function: sub_CA8D10
// Address: 0xca8d10
//
_BYTE *__fastcall sub_CA8D10(__int64 *a1, _BYTE *a2, __int64 a3, _QWORD *a4, __int64 a5, __int64 a6)
{
  _BOOL8 v7; // rbx
  unsigned __int64 v8; // rdx
  _BYTE *v9; // rbx
  unsigned __int64 v10; // r12
  __int64 v11; // rdi
  _QWORD *v12; // rdi
  _BYTE *result; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rax
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  unsigned int v29; // edi
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned int v49; // edi
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  unsigned int v54; // edi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  _QWORD v59[4]; // [rsp+10h] [rbp-90h] BYREF
  char v60; // [rsp+30h] [rbp-70h]
  char v61; // [rsp+31h] [rbp-6Fh]
  _BYTE *v62; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v63; // [rsp+48h] [rbp-58h]
  _BOOL8 v64; // [rsp+50h] [rbp-50h]
  _QWORD *v65; // [rsp+58h] [rbp-48h]
  __int64 v66; // [rsp+60h] [rbp-40h]
  _QWORD v67[7]; // [rsp+68h] [rbp-38h] BYREF

  if ( a3 == 1 )
  {
    v11 = *a1;
    v63 = a2;
    v65 = v67;
    v59[0] = "Unrecognized escape code";
    LODWORD(v62) = 0;
    v66 = 0;
    LOBYTE(v67[0]) = 0;
    v64 = 1;
    v61 = 1;
    v60 = 3;
    sub_CA8D00(v11, (__int64)v59, (__int64)&v62, (__int64)a4, a5);
    v12 = v65;
    result = 0;
    a4[1] = 0;
    if ( v12 != v67 )
    {
LABEL_4:
      j_j___libc_free_0(v12, v67[0] + 1LL);
      return 0;
    }
  }
  else
  {
    v7 = a3 != 0;
    v8 = a3 - v7;
    v9 = &a2[v7];
    v10 = v8;
    switch ( *v9 )
    {
      case 9:
      case 0x74:
        v15 = a4[1];
        if ( (unsigned __int64)(v15 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v15 + 1, 1u, a5, a6);
          v15 = a4[1];
        }
        *(_BYTE *)(*a4 + v15) = 9;
        ++a4[1];
        goto LABEL_11;
      case 0xA:
        goto LABEL_14;
      case 0xD:
        if ( v8 <= 1 )
        {
LABEL_14:
          if ( !v8 )
            goto LABEL_16;
        }
        else if ( v9[1] == 10 )
        {
          v10 = v8 - 1;
          ++v9;
        }
        --v10;
        ++v9;
LABEL_16:
        v62 = v9;
        v63 = (_BYTE *)v10;
        v16 = sub_C935B0(&v62, (unsigned __int8 *)" \t", 2, 0);
        v17 = (unsigned __int64)v63;
        if ( v16 < (unsigned __int64)v63 )
          v17 = v16;
        result = &v62[v17];
        break;
      case 0x20:
        v44 = a4[1];
        if ( (unsigned __int64)(v44 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v44 + 1, 1u, a5, a6);
          v44 = a4[1];
        }
        *(_BYTE *)(*a4 + v44) = 32;
        ++a4[1];
        goto LABEL_11;
      case 0x22:
        v58 = a4[1];
        if ( (unsigned __int64)(v58 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v58 + 1, 1u, a5, a6);
          v58 = a4[1];
        }
        *(_BYTE *)(*a4 + v58) = 34;
        ++a4[1];
        goto LABEL_11;
      case 0x2F:
        v41 = a4[1];
        if ( (unsigned __int64)(v41 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v41 + 1, 1u, a5, a6);
          v41 = a4[1];
        }
        *(_BYTE *)(*a4 + v41) = 47;
        ++a4[1];
        goto LABEL_11;
      case 0x30:
        v42 = a4[1];
        if ( (unsigned __int64)(v42 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v42 + 1, 1u, a5, a6);
          v42 = a4[1];
        }
        *(_BYTE *)(*a4 + v42) = 0;
        ++a4[1];
        goto LABEL_11;
      case 0x4C:
        v31 = a4[1];
        if ( (unsigned __int64)(v31 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v31 + 1, 1u, a5, a6);
          v31 = a4[1];
        }
        *(_BYTE *)(*a4 + v31) = -30;
        v32 = a4[1];
        v33 = v32 + 1;
        v34 = v32 + 2;
        a4[1] = v33;
        if ( v34 > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v34, 1u, a5, a6);
          v33 = a4[1];
        }
        *(_BYTE *)(*a4 + v33) = 0x80;
        v35 = a4[1];
        v36 = v35 + 1;
        v37 = v35 + 2;
        a4[1] = v36;
        if ( v37 > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v37, 1u, a5, a6);
          v36 = a4[1];
        }
        *(_BYTE *)(*a4 + v36) = -88;
        ++a4[1];
        goto LABEL_11;
      case 0x4E:
        sub_CA6D50(0x85u, a4, v8, (__int64)a4, a5, a6);
        goto LABEL_11;
      case 0x50:
        v18 = a4[1];
        if ( (unsigned __int64)(v18 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v18 + 1, 1u, a5, a6);
          v18 = a4[1];
        }
        *(_BYTE *)(*a4 + v18) = -30;
        v19 = a4[1];
        v20 = v19 + 1;
        v21 = v19 + 2;
        a4[1] = v20;
        if ( v21 > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v21, 1u, a5, a6);
          v20 = a4[1];
        }
        *(_BYTE *)(*a4 + v20) = 0x80;
        v22 = a4[1];
        v23 = v22 + 1;
        v24 = v22 + 2;
        a4[1] = v23;
        if ( v24 > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v24, 1u, a5, a6);
          v23 = a4[1];
        }
        *(_BYTE *)(*a4 + v23) = -87;
        ++a4[1];
        goto LABEL_11;
      case 0x55:
        if ( v8 <= 8 )
          goto LABEL_11;
        if ( sub_C93C90((__int64)(v9 + 1), 8, 0x10u, (unsigned __int64 *)&v62)
          || (v29 = (unsigned int)v62, v62 != (_BYTE *)(unsigned int)v62) )
        {
          v29 = 65533;
        }
        sub_CA6D50(v29, a4, v25, v26, v27, v28);
        return v9 + 9;
      case 0x5C:
        v30 = a4[1];
        if ( (unsigned __int64)(v30 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v30 + 1, 1u, a5, a6);
          v30 = a4[1];
        }
        *(_BYTE *)(*a4 + v30) = 92;
        ++a4[1];
        goto LABEL_11;
      case 0x5F:
        sub_CA6D50(0xA0u, a4, v8, (__int64)a4, a5, a6);
        goto LABEL_11;
      case 0x61:
        v40 = a4[1];
        if ( (unsigned __int64)(v40 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v40 + 1, 1u, a5, a6);
          v40 = a4[1];
        }
        *(_BYTE *)(*a4 + v40) = 7;
        ++a4[1];
        goto LABEL_11;
      case 0x62:
        v56 = a4[1];
        if ( (unsigned __int64)(v56 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v56 + 1, 1u, a5, a6);
          v56 = a4[1];
        }
        *(_BYTE *)(*a4 + v56) = 8;
        ++a4[1];
        goto LABEL_11;
      case 0x65:
        v43 = a4[1];
        if ( (unsigned __int64)(v43 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v43 + 1, 1u, a5, a6);
          v43 = a4[1];
        }
        *(_BYTE *)(*a4 + v43) = 27;
        ++a4[1];
        goto LABEL_11;
      case 0x66:
        v57 = a4[1];
        if ( (unsigned __int64)(v57 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v57 + 1, 1u, a5, a6);
          v57 = a4[1];
        }
        *(_BYTE *)(*a4 + v57) = 12;
        ++a4[1];
        goto LABEL_11;
      case 0x6E:
        v38 = a4[1];
        if ( (unsigned __int64)(v38 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v38 + 1, 1u, a5, a6);
          v38 = a4[1];
        }
        *(_BYTE *)(*a4 + v38) = 10;
        ++a4[1];
        goto LABEL_11;
      case 0x72:
        v39 = a4[1];
        if ( (unsigned __int64)(v39 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v39 + 1, 1u, a5, a6);
          v39 = a4[1];
        }
        *(_BYTE *)(*a4 + v39) = 13;
        ++a4[1];
        goto LABEL_11;
      case 0x75:
        if ( v8 <= 4 )
          goto LABEL_11;
        if ( sub_C93C90((__int64)(v9 + 1), 4, 0x10u, (unsigned __int64 *)&v62)
          || (v54 = (unsigned int)v62, v62 != (_BYTE *)(unsigned int)v62) )
        {
          v54 = 65533;
        }
        sub_CA6D50(v54, a4, v50, v51, v52, v53);
        return v9 + 5;
      case 0x76:
        v55 = a4[1];
        if ( (unsigned __int64)(v55 + 1) > a4[2] )
        {
          sub_C8D290((__int64)a4, a4 + 3, v55 + 1, 1u, a5, a6);
          v55 = a4[1];
        }
        *(_BYTE *)(*a4 + v55) = 11;
        ++a4[1];
        goto LABEL_11;
      case 0x78:
        if ( v8 <= 2 )
        {
LABEL_11:
          if ( v10 )
            ++v9;
          return v9;
        }
        else
        {
          if ( sub_C93C90((__int64)(v9 + 1), 2, 0x10u, (unsigned __int64 *)&v62)
            || (v49 = (unsigned int)v62, v62 != (_BYTE *)(unsigned int)v62) )
          {
            v49 = 65533;
          }
          sub_CA6D50(v49, a4, v45, v46, v47, v48);
          return v9 + 3;
        }
      default:
        v14 = *a1;
        LODWORD(v62) = 0;
        v59[0] = "Unrecognized escape code";
        v64 = v8 != 0;
        v65 = v67;
        v66 = 0;
        LOBYTE(v67[0]) = 0;
        v63 = v9;
        v61 = 1;
        v60 = 3;
        sub_CA8D00(v14, (__int64)v59, (__int64)&v62, (__int64)a4, (__int64)&v62);
        v12 = v65;
        result = 0;
        a4[1] = 0;
        if ( v12 != v67 )
          goto LABEL_4;
        return result;
    }
  }
  return result;
}
