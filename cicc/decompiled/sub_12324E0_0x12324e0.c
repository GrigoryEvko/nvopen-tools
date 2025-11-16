// Function: sub_12324E0
// Address: 0x12324e0
//
__int64 __fastcall sub_12324E0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r15
  __int64 v4; // r14
  unsigned int v6; // r12d
  __int64 *v7; // rbx
  int v8; // eax
  __int64 result; // rax
  char v10; // r10
  __int64 v11; // rcx
  int v12; // edx
  int v13; // edx
  __int64 *v14; // rdi
  char v15; // al
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 *v18; // r13
  _QWORD *v19; // rax
  char v20; // r10
  __int64 v21; // r15
  __int64 v22; // r11
  unsigned int v23; // ecx
  __int64 *v24; // rsi
  int v26; // eax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  int v30; // ecx
  char v31; // cl
  const char *v32; // rax
  __int64 *v33; // r8
  __int64 *v34; // rax
  __int64 v35; // rsi
  int v36; // edx
  const char *v37; // rax
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  bool v40; // zf
  int v41; // ecx
  char v42; // al
  char v43; // dl
  int v44; // eax
  __int64 v45; // rax
  unsigned __int64 v46; // [rsp+8h] [rbp-188h]
  __int64 v47; // [rsp+8h] [rbp-188h]
  char v48; // [rsp+18h] [rbp-178h]
  __int64 v49; // [rsp+18h] [rbp-178h]
  int v50; // [rsp+18h] [rbp-178h]
  char v51; // [rsp+24h] [rbp-16Ch]
  int v52; // [rsp+24h] [rbp-16Ch]
  char v53; // [rsp+24h] [rbp-16Ch]
  char v54; // [rsp+24h] [rbp-16Ch]
  int v55; // [rsp+28h] [rbp-168h]
  char v56; // [rsp+28h] [rbp-168h]
  char v57; // [rsp+28h] [rbp-168h]
  __int64 *v58; // [rsp+28h] [rbp-168h]
  char v59; // [rsp+28h] [rbp-168h]
  unsigned __int64 v60; // [rsp+38h] [rbp-158h]
  __int64 v61; // [rsp+38h] [rbp-158h]
  char v62; // [rsp+38h] [rbp-158h]
  unsigned int v63; // [rsp+38h] [rbp-158h]
  unsigned int v64; // [rsp+38h] [rbp-158h]
  __int64 v65; // [rsp+40h] [rbp-150h] BYREF
  __int64 v66; // [rsp+48h] [rbp-148h] BYREF
  __int64 *v67; // [rsp+50h] [rbp-140h] BYREF
  __int64 v68; // [rsp+58h] [rbp-138h]
  _QWORD v69[4]; // [rsp+60h] [rbp-130h] BYREF
  __int16 v70; // [rsp+80h] [rbp-110h]
  const char *v71; // [rsp+90h] [rbp-100h] BYREF
  char *v72; // [rsp+98h] [rbp-F8h]
  __int64 v73; // [rsp+A0h] [rbp-F0h]
  int v74; // [rsp+A8h] [rbp-E8h]
  char v75; // [rsp+ACh] [rbp-E4h]
  char v76; // [rsp+B0h] [rbp-E0h] BYREF
  char v77; // [rsp+B1h] [rbp-DFh]
  const char *v78; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v79; // [rsp+D8h] [rbp-B8h]
  _BYTE v80[16]; // [rsp+E0h] [rbp-B0h] BYREF
  char v81; // [rsp+F0h] [rbp-A0h]
  char v82; // [rsp+F1h] [rbp-9Fh]

  v3 = a1;
  v4 = a1 + 176;
  v6 = 0;
  v7 = a2;
  v65 = 0;
  v66 = 0;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v8 = *(_DWORD *)(a1 + 240);
        if ( v8 != 90 )
          break;
        v6 |= 3u;
        *(_DWORD *)(a1 + 240) = sub_1205200(v4);
      }
      if ( v8 != 87 )
        break;
      v6 |= 2u;
      *(_DWORD *)(a1 + 240) = sub_1205200(v4);
    }
    if ( v8 != 85 )
      break;
    v6 |= 4u;
    *(_DWORD *)(a1 + 240) = sub_1205200(v4);
  }
  v67 = 0;
  v82 = 1;
  v78 = "expected type";
  v81 = 3;
  if ( (unsigned __int8)sub_12190A0(a1, &v67, (int *)&v78, 0) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected comma after getelementptr's type") )
    return 1;
  v60 = *(_QWORD *)(a1 + 232);
  v10 = sub_122FE20((__int64 **)a1, &v65, a3);
  if ( v10 )
    return 1;
  v11 = *(_QWORD *)(v65 + 8);
  v12 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v12 - 17) <= 1 )
    LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( (_BYTE)v12 != 14 )
  {
    v82 = 1;
    v78 = "base of getelementptr must be a pointer";
    v81 = 3;
    sub_11FD800(a1 + 176, v60, (__int64)&v78, 1);
    return 1;
  }
  v78 = v80;
  v79 = 0x1000000000LL;
  v13 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v13 - 17) > 1 )
  {
    v51 = 0;
    v55 = 0;
  }
  else
  {
    v51 = (_BYTE)v13 == 18;
    v55 = *(_DWORD *)(v11 + 32);
  }
  if ( *(_DWORD *)(a1 + 240) != 4 )
  {
    v71 = 0;
    v14 = v67;
    v72 = &v76;
    v73 = 4;
    v74 = 0;
    v75 = 1;
LABEL_19:
    v15 = *((_BYTE *)v14 + 8);
    goto LABEL_20;
  }
  v49 = a1 + 176;
  while ( 1 )
  {
    v26 = sub_1205200(v49);
    *(_DWORD *)(a1 + 240) = v26;
    if ( v26 == 511 )
      break;
    v24 = &v66;
    v46 = *(_QWORD *)(a1 + 232);
    v10 = sub_122FE20((__int64 **)a1, &v66, a3);
    if ( v10 )
    {
      result = 1;
      goto LABEL_30;
    }
    v28 = v66;
    v29 = *(_QWORD *)(v66 + 8);
    v30 = *(unsigned __int8 *)(v29 + 8);
    if ( (unsigned int)(v30 - 17) > 1 )
    {
      if ( (_BYTE)v30 != 12 )
      {
LABEL_76:
        v77 = 1;
        v32 = "getelementptr index must be an integer";
        goto LABEL_43;
      }
    }
    else
    {
      if ( *(_BYTE *)(**(_QWORD **)(v29 + 16) + 8LL) != 12 )
        goto LABEL_76;
      v31 = (_BYTE)v30 == 18;
      if ( v55 || v51 )
      {
        if ( *(_DWORD *)(v29 + 32) != v55 || v31 != v51 )
        {
          v77 = 1;
          v32 = "getelementptr vector index has a wrong number of elements";
LABEL_43:
          v24 = (__int64 *)v46;
          v71 = v32;
          v76 = 3;
          sub_11FD800(v49, v46, (__int64)&v71, 1);
          result = 1;
          goto LABEL_30;
        }
      }
      else
      {
        v51 = v31;
        v55 = *(_DWORD *)(v29 + 32);
      }
    }
    v38 = (unsigned int)v79;
    v39 = (unsigned int)v79 + 1LL;
    if ( v39 > HIDWORD(v79) )
    {
      v47 = v66;
      sub_C8D5F0((__int64)&v78, v80, v39, 8u, v27, v66);
      v38 = (unsigned int)v79;
      v10 = 0;
      v28 = v47;
    }
    *(_QWORD *)&v78[8 * v38] = v28;
    v40 = *(_DWORD *)(a1 + 240) == 4;
    v16 = (unsigned int)(v79 + 1);
    LODWORD(v79) = v79 + 1;
    if ( !v40 )
    {
      v7 = a2;
      v3 = a1;
      goto LABEL_58;
    }
  }
  v7 = a2;
  v16 = (unsigned int)v79;
  v3 = a1;
  v10 = 1;
LABEL_58:
  v71 = 0;
  v14 = v67;
  v72 = &v76;
  v73 = 4;
  v74 = 0;
  v75 = 1;
  if ( !(_DWORD)v16 )
    goto LABEL_19;
  v41 = *((unsigned __int8 *)v67 + 8);
  v15 = *((_BYTE *)v67 + 8);
  if ( (_BYTE)v41 == 12 )
    goto LABEL_23;
  if ( (unsigned __int8)v41 > 3u )
  {
    if ( (_BYTE)v41 == 5 )
      goto LABEL_44;
    if ( (v41 & 0xFD) != 4 && (v41 & 0xFB) != 0xA )
    {
      if ( (unsigned __int8)(v15 - 15) > 3u && v41 != 20
        || (v59 = v10, v42 = sub_BCEBA0((__int64)v67, (__int64)&v71), v10 = v59, !v42) )
      {
        v24 = (__int64 *)v60;
        v69[0] = "base element of getelementptr must be sized";
        v70 = 259;
        sub_11FD800(v49, v60, (__int64)v69, 1);
        result = 1;
        goto LABEL_28;
      }
      v14 = v67;
      goto LABEL_19;
    }
  }
LABEL_20:
  if ( v15 == 15 )
  {
    v56 = v10;
    if ( sub_BCEA30((__int64)v14) )
    {
      HIBYTE(v70) = 1;
      v37 = "getelementptr cannot target structure that contains scalable vector type";
      goto LABEL_52;
    }
    v14 = v67;
    v16 = (unsigned int)v79;
    v10 = v56;
  }
  else
  {
LABEL_44:
    v16 = (unsigned int)v79;
  }
LABEL_23:
  v57 = v10;
  if ( !sub_B4DC50((__int64)v14, (__int64)v78, v16) )
  {
    HIBYTE(v70) = 1;
    v37 = "invalid getelementptr indices";
LABEL_52:
    v24 = (__int64 *)v60;
    v69[0] = v37;
    LOBYTE(v70) = 3;
    sub_11FD800(v3 + 176, v60, (__int64)v69, 1);
    result = 1;
    goto LABEL_28;
  }
  v17 = (unsigned int)v79;
  v70 = 257;
  v48 = v57;
  v18 = (__int64 *)v78;
  v61 = v65;
  v58 = v67;
  v52 = v79 + 1;
  v19 = sub_BD2C40(88, (int)v79 + 1);
  v20 = v48;
  v21 = (__int64)v19;
  if ( v19 )
  {
    v22 = *(_QWORD *)(v61 + 8);
    v23 = v52 & 0x7FFFFFF;
    if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 > 1 )
    {
      v33 = &v18[v17];
      if ( v18 != v33 )
      {
        v34 = v18;
        while ( 1 )
        {
          v35 = *(_QWORD *)(*v34 + 8);
          v36 = *(unsigned __int8 *)(v35 + 8);
          if ( v36 == 17 )
          {
            v43 = 0;
            goto LABEL_70;
          }
          if ( v36 == 18 )
            break;
          if ( v33 == ++v34 )
            goto LABEL_26;
        }
        v43 = 1;
LABEL_70:
        v44 = *(_DWORD *)(v35 + 32);
        BYTE4(v68) = v43;
        v50 = v52 & 0x7FFFFFF;
        LODWORD(v68) = v44;
        v54 = v20;
        v45 = sub_BCE1B0((__int64 *)v22, v68);
        v20 = v54;
        v23 = v50;
        v22 = v45;
      }
    }
LABEL_26:
    v53 = v20;
    sub_B44260(v21, v22, 34, v23, 0, 0);
    *(_QWORD *)(v21 + 72) = v58;
    *(_QWORD *)(v21 + 80) = sub_B4DC50((__int64)v58, (__int64)v18, v17);
    sub_B4D9A0(v21, v61, v18, v17, (__int64)v69);
    v20 = v53;
  }
  *v7 = v21;
  v24 = (__int64 *)v6;
  v62 = v20;
  sub_B4DDE0(v21, v6);
  result = 2 * (unsigned int)(v62 != 0);
LABEL_28:
  if ( !v75 )
  {
    v64 = result;
    _libc_free(v72, v24);
    result = v64;
  }
LABEL_30:
  if ( v78 != v80 )
  {
    v63 = result;
    _libc_free(v78, v24);
    return v63;
  }
  return result;
}
