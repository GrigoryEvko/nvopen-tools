// Function: sub_3855D50
// Address: 0x3855d50
//
__int64 __fastcall sub_3855D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  int v8; // r11d
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // r11d
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  unsigned int v21; // r10d
  int v23; // eax
  unsigned int v24; // r10d
  __int64 *v25; // rsi
  __int64 v26; // rbx
  _QWORD *v27; // rax
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rdx
  unsigned int v31; // eax
  __int64 *v32; // rsi
  __int64 v33; // rdi
  __int64 v34; // rdx
  __int64 *v35; // rax
  bool v36; // cc
  int v37; // edx
  _QWORD *v38; // rax
  unsigned int v39; // esi
  __int64 *v40; // rdx
  bool v41; // r10
  int v42; // eax
  int v43; // r10d
  __int64 v44; // rdx
  int v45; // esi
  unsigned int v46; // edi
  __int64 *v47; // rax
  __int64 v48; // r11
  unsigned int v49; // edi
  __int64 *v50; // rax
  __int64 v51; // r10
  __int64 v52; // rdx
  _QWORD *v53; // rax
  int v54; // esi
  int v55; // edx
  __int64 *v56; // rax
  int v57; // esi
  _QWORD *v58; // rax
  unsigned __int64 v59; // rdi
  unsigned int v60; // edx
  unsigned __int64 v61; // rsi
  int v62; // esi
  int v63; // r8d
  __int64 v64; // rax
  int v65; // eax
  int v66; // ecx
  int v67; // eax
  int v68; // ecx
  int v69; // r8d
  int v70; // r8d
  unsigned __int64 v71; // rdi
  unsigned int v72; // edx
  __int64 v73; // rax
  __int64 v74; // [rsp+0h] [rbp-C0h]
  bool v75; // [rsp+0h] [rbp-C0h]
  __int64 *v76; // [rsp+8h] [rbp-B8h]
  char v77; // [rsp+17h] [rbp-A9h]
  __int64 v78; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v79; // [rsp+18h] [rbp-A8h]
  unsigned __int8 v80; // [rsp+18h] [rbp-A8h]
  __int64 v81; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v82; // [rsp+28h] [rbp-98h] BYREF
  __int64 v83; // [rsp+30h] [rbp-90h] BYREF
  const void *v84; // [rsp+38h] [rbp-88h] BYREF
  unsigned int v85; // [rsp+40h] [rbp-80h]
  __int64 v86; // [rsp+50h] [rbp-70h]
  const void *v87; // [rsp+58h] [rbp-68h] BYREF
  unsigned int v88; // [rsp+60h] [rbp-60h]
  __int64 v89; // [rsp+70h] [rbp-50h] BYREF
  __int64 v90; // [rsp+78h] [rbp-48h]
  __int64 v91; // [rsp+80h] [rbp-40h]
  __int64 v92; // [rsp+88h] [rbp-38h]

  v6 = *(_QWORD *)(a2 - 48);
  v7 = *(_QWORD *)(a2 - 24);
  v8 = *(_DWORD *)(a1 + 160);
  v9 = *(_BYTE *)(v7 + 16);
  v77 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  v10 = a1 + 136;
  v11 = *(_QWORD *)(a1 + 144);
  v78 = v10;
  if ( *(_BYTE *)(v6 + 16) > 0x10u )
  {
    if ( !v8 )
      goto LABEL_44;
    v23 = v8 - 1;
    v13 = v23;
    v24 = v23 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v25 = (__int64 *)(v11 + 16LL * v24);
    a4 = *v25;
    if ( v6 == *v25 )
    {
LABEL_13:
      a4 = v25[1];
      v76 = (__int64 *)a4;
    }
    else
    {
      v54 = 1;
      while ( a4 != -8 )
      {
        v69 = v54 + 1;
        v24 = v23 & (v54 + v24);
        v25 = (__int64 *)(v11 + 16LL * v24);
        a4 = *v25;
        if ( v6 == *v25 )
          goto LABEL_13;
        v54 = v69;
      }
      v76 = 0;
    }
    if ( v9 <= 0x10u )
    {
      v74 = v7;
      v12 = *(_QWORD *)(a2 - 72);
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( v9 > 0x10u )
  {
    if ( !v8 )
      goto LABEL_44;
    v76 = *(__int64 **)(a2 - 48);
    v23 = v8 - 1;
LABEL_38:
    v13 = v23;
    v39 = v23 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v40 = (__int64 *)(v11 + 16LL * v39);
    a4 = *v40;
    if ( v7 == *v40 )
    {
LABEL_39:
      v74 = v40[1];
    }
    else
    {
      v55 = 1;
      while ( a4 != -8 )
      {
        v70 = v55 + 1;
        v39 = v23 & (v55 + v39);
        v40 = (__int64 *)(v11 + 16LL * v39);
        a4 = *v40;
        if ( v7 == *v40 )
          goto LABEL_39;
        v55 = v70;
      }
      v74 = 0;
    }
    v12 = *(_QWORD *)(a2 - 72);
    goto LABEL_5;
  }
  v12 = *(_QWORD *)(a2 - 72);
  if ( !v8 )
  {
    v76 = (__int64 *)v6;
    v41 = v6 == v7;
    goto LABEL_43;
  }
  v74 = v7;
  v13 = v8 - 1;
  v76 = (__int64 *)v6;
LABEL_5:
  v14 = v13 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v15 = (__int64 *)(v11 + 16 * v14);
  v16 = *v15;
  if ( *v15 != v12 )
  {
    v42 = 1;
    while ( v16 != -8 )
    {
      a4 = (unsigned int)(v42 + 1);
      v14 = v13 & (unsigned int)(v42 + v14);
      v15 = (__int64 *)(v11 + 16LL * (unsigned int)v14);
      v16 = *v15;
      if ( *v15 == v12 )
        goto LABEL_6;
      v42 = a4;
    }
    v41 = v76 == (__int64 *)v74 && v76 != 0;
LABEL_43:
    v75 = v41;
    if ( v41 )
    {
      v89 = a2;
      v53 = sub_38526A0(v78, &v89);
      v21 = v75;
      v53[1] = v76;
      return v21;
    }
LABEL_44:
    if ( v77 != 15 )
      return sub_384F9A0(a1, a2);
    v43 = *(_DWORD *)(a1 + 256);
    v44 = *(_QWORD *)(a1 + 240);
    if ( v43 )
    {
      v45 = v43 - 1;
      v46 = (v43 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v47 = (__int64 *)(v44 + 32LL * v46);
      v48 = *v47;
      if ( v6 == *v47 )
      {
LABEL_51:
        v83 = v47[1];
        v85 = *((_DWORD *)v47 + 6);
        if ( v85 <= 0x40 )
        {
          v84 = (const void *)v47[2];
          goto LABEL_53;
        }
        sub_16A4FD0((__int64)&v84, (const void **)v47 + 2);
        v44 = *(_QWORD *)(a1 + 240);
        v43 = *(_DWORD *)(a1 + 256);
LABEL_67:
        if ( !v43 )
          goto LABEL_69;
        v45 = v43 - 1;
LABEL_53:
        v49 = v45 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v50 = (__int64 *)(v44 + 32LL * v49);
        v51 = *v50;
        if ( v7 == *v50 )
        {
LABEL_54:
          v52 = v50[1];
          v86 = v52;
          v88 = *((_DWORD *)v50 + 6);
          if ( v88 > 0x40 )
          {
            sub_16A4FD0((__int64)&v87, (const void **)v50 + 2);
            v52 = v86;
          }
          else
          {
            v87 = (const void *)v50[2];
          }
LABEL_56:
          if ( v83 == v52 )
          {
            if ( v85 <= 0x40 )
            {
              if ( v84 != v87 )
                goto LABEL_57;
            }
            else if ( !sub_16A5220((__int64)&v84, &v87) )
            {
              goto LABEL_57;
            }
            if ( v83 )
            {
              v89 = a2;
              v56 = sub_3854530(a1 + 232, &v89);
              v36 = *((_DWORD *)v56 + 6) <= 0x40u;
              v56[1] = v83;
              if ( v36 && v85 <= 0x40 )
              {
                v71 = (unsigned __int64)v84;
                v56[2] = (__int64)v84;
                v72 = v85;
                *((_DWORD *)v56 + 6) = v85;
                if ( v72 > 0x40 )
                {
                  v73 = (unsigned int)(((unsigned __int64)v72 + 63) >> 6) - 1;
                  *(_QWORD *)(v71 + 8 * v73) &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v72;
                }
                else
                {
                  v56[2] = v71 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v72);
                }
              }
              else
              {
                sub_16A51C0((__int64)(v56 + 2), (__int64)&v84);
              }
              v57 = *(_DWORD *)(a1 + 184);
              v89 = 0;
              v90 = -1;
              v91 = 0;
              v92 = 0;
              if ( v57 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, v6, &v81, &v89) )
              {
                v82 = a2;
                v58 = sub_176FB00(a1 + 168, &v82);
                v58[1] = v81;
              }
              v21 = 1;
              goto LABEL_58;
            }
          }
LABEL_57:
          v21 = sub_384F9A0(a1, a2);
LABEL_58:
          if ( v88 > 0x40 && v87 )
          {
            v79 = v21;
            j_j___libc_free_0_0((unsigned __int64)v87);
            v21 = v79;
          }
          if ( v85 > 0x40 && v84 )
          {
            v80 = v21;
            j_j___libc_free_0_0((unsigned __int64)v84);
            return v80;
          }
          return v21;
        }
        v67 = 1;
        while ( v51 != -8 )
        {
          v68 = v67 + 1;
          v49 = v45 & (v67 + v49);
          v50 = (__int64 *)(v44 + 32LL * v49);
          v51 = *v50;
          if ( v7 == *v50 )
            goto LABEL_54;
          v67 = v68;
        }
LABEL_69:
        v86 = 0;
        v52 = 0;
        v88 = 1;
        v87 = 0;
        goto LABEL_56;
      }
      v65 = 1;
      while ( v48 != -8 )
      {
        v66 = v65 + 1;
        v46 = v45 & (v65 + v46);
        v47 = (__int64 *)(v44 + 32LL * v46);
        v48 = *v47;
        if ( v6 == *v47 )
          goto LABEL_51;
        v65 = v66;
      }
    }
    v83 = 0;
    v85 = 1;
    v84 = 0;
    goto LABEL_67;
  }
LABEL_6:
  v17 = v15[1];
  if ( !v17 )
  {
    v41 = v76 != 0 && v76 == (__int64 *)v74;
    goto LABEL_43;
  }
  if ( sub_1596070(v15[1], v12, v14, a4) )
    goto LABEL_8;
  if ( sub_1593BB0(v17, v12, v18, v19) )
  {
    v6 = v7;
LABEL_8:
    if ( *(_BYTE *)(v6 + 16) <= 0x10u )
    {
      v89 = a2;
      v20 = sub_38526A0(v78, &v89);
      v21 = 1;
      v20[1] = v6;
      return v21;
    }
    v21 = 1;
    if ( v77 != 15 )
      return v21;
    v28 = *(_DWORD *)(a1 + 256);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 240);
      v31 = (v28 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v32 = (__int64 *)(v30 + 32LL * v31);
      v33 = *v32;
      if ( v6 != *v32 )
      {
        v62 = 1;
        while ( v33 != -8 )
        {
          v63 = v62 + 1;
          v31 = v29 & (v62 + v31);
          v32 = (__int64 *)(v30 + 32LL * v31);
          v33 = *v32;
          if ( v6 == *v32 )
            goto LABEL_23;
          v62 = v63;
        }
        return 1;
      }
LABEL_23:
      v34 = v32[1];
      v86 = v34;
      v88 = *((_DWORD *)v32 + 6);
      if ( v88 > 0x40 )
      {
        sub_16A4FD0((__int64)&v87, (const void **)v32 + 2);
        if ( !v86 )
        {
LABEL_32:
          if ( v88 > 0x40 )
          {
            if ( v87 )
              j_j___libc_free_0_0((unsigned __int64)v87);
          }
          return 1;
        }
LABEL_25:
        v89 = a2;
        v35 = sub_3854530(a1 + 232, &v89);
        v36 = *((_DWORD *)v35 + 6) <= 0x40u;
        v35[1] = v86;
        if ( v36 && v88 <= 0x40 )
        {
          v59 = (unsigned __int64)v87;
          v35[2] = (__int64)v87;
          v60 = v88;
          *((_DWORD *)v35 + 6) = v88;
          v61 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v60;
          if ( v60 > 0x40 )
          {
            v64 = (unsigned int)(((unsigned __int64)v60 + 63) >> 6) - 1;
            *(_QWORD *)(v59 + 8 * v64) &= v61;
          }
          else
          {
            v35[2] = v59 & v61;
          }
        }
        else
        {
          sub_16A51C0((__int64)(v35 + 2), (__int64)&v87);
        }
        v37 = *(_DWORD *)(a1 + 184);
        v89 = 0;
        v90 = -1;
        v91 = 0;
        v92 = 0;
        if ( v37 && *(_DWORD *)(a1 + 216) && sub_384F1D0(a1, v6, &v82, &v89) )
        {
          v83 = a2;
          v38 = sub_176FB00(a1 + 168, &v83);
          v38[1] = v82;
        }
        goto LABEL_32;
      }
      v87 = (const void *)v32[2];
      if ( v34 )
        goto LABEL_25;
    }
    return 1;
  }
  if ( v74 != 0 && v76 != 0 )
  {
    v26 = sub_15A2DC0(v17, v76, v74, 0);
    if ( v26 )
    {
      v89 = a2;
      v27 = sub_38526A0(v78, &v89);
      v21 = v74 != 0 && v76 != 0;
      v27[1] = v26;
      return v21;
    }
  }
  return sub_384F9A0(a1, a2);
}
