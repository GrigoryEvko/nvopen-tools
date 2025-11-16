// Function: sub_1884660
// Address: 0x1884660
//
__int64 __fastcall sub_1884660(__int64 a1, _BYTE *a2, __int64 a3, _QWORD *a4)
{
  char *v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // rbx
  __int64 *v10; // r8
  __int64 *v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // r14
  unsigned __int64 *v19; // r15
  unsigned __int64 *v20; // r13
  _QWORD *v21; // r14
  __int64 *v22; // r12
  unsigned __int64 *v23; // rbx
  __int64 *v24; // rax
  unsigned __int64 v25; // rcx
  __int64 *v26; // r9
  __int64 *v27; // rdx
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 *v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rsi
  char *v33; // rsi
  unsigned __int64 v34; // rdx
  char *v35; // rcx
  char *v36; // rdx
  __int64 v37; // rsi
  unsigned __int64 v38; // r13
  int v39; // r10d
  char v40; // al
  char v41; // di
  char v42; // al
  __int64 v43; // rdi
  char *v44; // rax
  char *v45; // rcx
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  const char *v50; // r13
  __int64 v51; // r15
  __int64 v52; // rdi
  __int64 v53; // r13
  __int64 v54; // r15
  __int64 v55; // rdi
  char *v56; // rsi
  void (__fastcall *v58)(__int64, const char **); // rax
  __int64 v59; // [rsp+10h] [rbp-1A0h]
  __int64 *v60; // [rsp+30h] [rbp-180h]
  __int64 v61; // [rsp+40h] [rbp-170h]
  __int64 v62; // [rsp+40h] [rbp-170h]
  int v63; // [rsp+4Ch] [rbp-164h]
  const char *v64; // [rsp+58h] [rbp-158h] BYREF
  __int64 v65; // [rsp+60h] [rbp-150h] BYREF
  __int64 v66; // [rsp+68h] [rbp-148h]
  __int64 v67; // [rsp+70h] [rbp-140h]
  char *v68; // [rsp+80h] [rbp-130h] BYREF
  char *v69; // [rsp+88h] [rbp-128h]
  char *v70; // [rsp+90h] [rbp-120h]
  char *v71; // [rsp+A0h] [rbp-110h] BYREF
  char *v72; // [rsp+A8h] [rbp-108h]
  char *v73; // [rsp+B0h] [rbp-100h]
  _QWORD v74[2]; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-E0h]
  _QWORD v76[2]; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v77; // [rsp+F0h] [rbp-C0h]
  __int64 v78[2]; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+110h] [rbp-A0h]
  __int64 v80[2]; // [rsp+120h] [rbp-90h] BYREF
  __int64 v81; // [rsp+130h] [rbp-80h]
  __int64 v82; // [rsp+140h] [rbp-70h] BYREF
  __int64 v83; // [rsp+148h] [rbp-68h]
  __int64 v84; // [rsp+150h] [rbp-60h]
  const char *v85; // [rsp+160h] [rbp-50h] BYREF
  const char *v86; // [rsp+168h] [rbp-48h]
  _QWORD v87[8]; // [rsp+170h] [rbp-40h] BYREF

  v65 = 0;
  v66 = 0;
  v67 = 0;
  if ( a2 )
  {
    v85 = (const char *)v87;
    sub_18736F0((__int64 *)&v85, a2, (__int64)&a2[a3]);
    v7 = (char *)v85;
  }
  else
  {
    v86 = 0;
    v85 = (const char *)v87;
    v7 = (char *)v87;
    LOBYTE(v87[0]) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         v7,
         1,
         0,
         v80) )
  {
    sub_18843B0(a1, &v65);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v82);
  }
  if ( v85 != (const char *)v87 )
    j_j___libc_free_0(v85, v87[0] + 1LL);
  if ( sub_16D2B80((__int64)a2, a3, 0, (unsigned __int64 *)&v85) )
  {
    v58 = *(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 232LL);
    v85 = "key not an integer";
    LOWORD(v87[0]) = 259;
    v58(a1, &v85);
    return sub_187D570(&v65);
  }
  else
  {
    v8 = (__int64 *)a4[2];
    v9 = a4 + 1;
    v64 = v85;
    if ( v8 )
    {
      v10 = a4 + 1;
      v11 = v8;
      do
      {
        while ( 1 )
        {
          v12 = v11[2];
          v13 = v11[3];
          if ( (unsigned __int64)v85 <= v11[4] )
            break;
          v11 = (__int64 *)v11[3];
          if ( !v13 )
            goto LABEL_13;
        }
        v10 = v11;
        v11 = (__int64 *)v11[2];
      }
      while ( v12 );
LABEL_13:
      if ( v9 != v10 && (unsigned __int64)v85 >= v10[4] )
        goto LABEL_16;
    }
    LOBYTE(v85) = 0;
    sub_187DAA0(a4, (unsigned __int64 *)&v64, &v85);
    v8 = (__int64 *)a4[2];
    if ( v8 )
    {
LABEL_16:
      v14 = a4 + 1;
      do
      {
        while ( 1 )
        {
          v15 = v8[2];
          v16 = v8[3];
          if ( v8[4] >= (unsigned __int64)v64 )
            break;
          v8 = (__int64 *)v8[3];
          if ( !v16 )
            goto LABEL_20;
        }
        v14 = v8;
        v8 = (__int64 *)v8[2];
      }
      while ( v15 );
LABEL_20:
      v60 = v14;
      v17 = v14;
      if ( v9 != v14 )
      {
        if ( (unsigned __int64)v64 < v14[4] )
          v17 = a4 + 1;
        v60 = v17;
      }
    }
    else
    {
      v60 = a4 + 1;
    }
    v18 = v65;
    v59 = v66;
    if ( v65 != v66 )
    {
      while ( 1 )
      {
        v68 = 0;
        v69 = 0;
        v70 = 0;
        v19 = *(unsigned __int64 **)(v18 + 8);
        v20 = *(unsigned __int64 **)(v18 + 16);
        if ( v20 != v19 )
        {
          v61 = v18;
          v21 = a4;
          v22 = v9;
          v23 = v19;
          while ( 1 )
          {
            v24 = (__int64 *)v21[2];
            if ( v24 )
            {
              v25 = *v23;
              v26 = v22;
              v27 = (__int64 *)v21[2];
              do
              {
                while ( 1 )
                {
                  v28 = v27[2];
                  v29 = v27[3];
                  if ( v27[4] >= v25 )
                    break;
                  v27 = (__int64 *)v27[3];
                  if ( !v29 )
                    goto LABEL_32;
                }
                v26 = v27;
                v27 = (__int64 *)v27[2];
              }
              while ( v28 );
LABEL_32:
              if ( v22 != v26 && v25 >= v26[4] )
                goto LABEL_36;
            }
            LOBYTE(v85) = 0;
            sub_187DAA0(v21, v23, &v85);
            v24 = (__int64 *)v21[2];
            if ( v24 )
              break;
            v30 = v22;
LABEL_43:
            v33 = v69;
            v34 = (unsigned __int64)(v30 + 4) & 0xFFFFFFFFFFFFFFFBLL;
            v85 = (const char *)v34;
            if ( v69 == v70 )
            {
              sub_14F4870(&v68, v69, &v85);
            }
            else
            {
              if ( v69 )
              {
                *(_QWORD *)v69 = v34;
                v33 = v69;
              }
              v69 = v33 + 8;
            }
            if ( v20 == ++v23 )
            {
              v35 = v69;
              v36 = v68;
              v9 = v22;
              a4 = v21;
              v18 = v61;
              v37 = v69 - v68;
              v38 = v69 - v68;
              goto LABEL_49;
            }
          }
          v25 = *v23;
LABEL_36:
          v30 = v22;
          do
          {
            while ( 1 )
            {
              v31 = v24[2];
              v32 = v24[3];
              if ( v24[4] >= v25 )
                break;
              v24 = (__int64 *)v24[3];
              if ( !v32 )
                goto LABEL_40;
            }
            v30 = v24;
            v24 = (__int64 *)v24[2];
          }
          while ( v31 );
LABEL_40:
          if ( v22 != v30 && v25 < v30[4] )
            v30 = v22;
          goto LABEL_43;
        }
        v38 = 0;
        v37 = 0;
        v36 = 0;
        v35 = 0;
LABEL_49:
        v39 = v63;
        v40 = (32 * *(_BYTE *)(v18 + 5)) | *(_DWORD *)v18 & 0xF | (16 * *(_BYTE *)(v18 + 4));
        v41 = *(_BYTE *)(v18 + 6);
        v71 = 0;
        v72 = 0;
        v73 = 0;
        v42 = (v41 << 6) | v40;
        v43 = v63 & 0xFFFFFF80;
        LOBYTE(v39) = v63 & 0x80 | v42 & 0x7F;
        v63 = v39;
        if ( v37 )
        {
          if ( v38 > 0x7FFFFFFFFFFFFFF8LL )
            sub_4261EA(v43, v37, v36);
          v44 = (char *)sub_22077B0(v38);
          v35 = v69;
          v36 = v68;
        }
        else
        {
          v44 = 0;
        }
        v71 = v44;
        v72 = v44;
        v73 = &v44[v38];
        if ( v35 == v36 )
        {
          v45 = v44;
        }
        else
        {
          v45 = &v44[v35 - v36];
          do
          {
            if ( v44 )
              *(_QWORD *)v44 = *(_QWORD *)v36;
            v44 += 8;
            v36 += 8;
          }
          while ( v44 != v45 );
        }
        v72 = v45;
        v74[0] = 0;
        v74[1] = 0;
        v75 = 0;
        v76[0] = *(_QWORD *)(v18 + 32);
        v76[1] = *(_QWORD *)(v18 + 40);
        v77 = *(_QWORD *)(v18 + 48);
        v46 = *(_QWORD *)(v18 + 56);
        *(_QWORD *)(v18 + 48) = 0;
        *(_QWORD *)(v18 + 40) = 0;
        *(_QWORD *)(v18 + 32) = 0;
        v78[0] = v46;
        v78[1] = *(_QWORD *)(v18 + 64);
        v79 = *(_QWORD *)(v18 + 72);
        v47 = *(_QWORD *)(v18 + 80);
        *(_QWORD *)(v18 + 72) = 0;
        *(_QWORD *)(v18 + 64) = 0;
        *(_QWORD *)(v18 + 56) = 0;
        v80[0] = v47;
        v80[1] = *(_QWORD *)(v18 + 88);
        v81 = *(_QWORD *)(v18 + 96);
        v48 = *(_QWORD *)(v18 + 104);
        *(_QWORD *)(v18 + 96) = 0;
        *(_QWORD *)(v18 + 88) = 0;
        *(_QWORD *)(v18 + 80) = 0;
        v82 = v48;
        v83 = *(_QWORD *)(v18 + 112);
        v84 = *(_QWORD *)(v18 + 120);
        v49 = *(_QWORD *)(v18 + 128);
        *(_QWORD *)(v18 + 120) = 0;
        *(_QWORD *)(v18 + 112) = 0;
        *(_QWORD *)(v18 + 104) = 0;
        v85 = (const char *)v49;
        v86 = *(const char **)(v18 + 136);
        v87[0] = *(_QWORD *)(v18 + 144);
        *(_QWORD *)(v18 + 144) = 0;
        *(_QWORD *)(v18 + 136) = 0;
        *(_QWORD *)(v18 + 128) = 0;
        v62 = sub_22077B0(104);
        if ( v62 )
          sub_142CF20(v62, v63, 0, 0, (__int64 *)&v71, v74, v76, v78, v80, &v82, &v85);
        v50 = v86;
        v51 = (__int64)v85;
        if ( v86 != v85 )
        {
          do
          {
            v52 = *(_QWORD *)(v51 + 16);
            if ( v52 )
              j_j___libc_free_0(v52, *(_QWORD *)(v51 + 32) - v52);
            v51 += 40;
          }
          while ( v50 != (const char *)v51 );
          v51 = (__int64)v85;
        }
        if ( v51 )
          j_j___libc_free_0(v51, v87[0] - v51);
        v53 = v83;
        v54 = v82;
        if ( v83 != v82 )
        {
          do
          {
            v55 = *(_QWORD *)(v54 + 16);
            if ( v55 )
              j_j___libc_free_0(v55, *(_QWORD *)(v54 + 32) - v55);
            v54 += 40;
          }
          while ( v53 != v54 );
          v54 = v82;
        }
        if ( v54 )
          j_j___libc_free_0(v54, v84 - v54);
        if ( v80[0] )
          j_j___libc_free_0(v80[0], v81 - v80[0]);
        if ( v78[0] )
          j_j___libc_free_0(v78[0], v79 - v78[0]);
        if ( v76[0] )
          j_j___libc_free_0(v76[0], v77 - v76[0]);
        if ( v74[0] )
          j_j___libc_free_0(v74[0], v75 - v74[0]);
        if ( v71 )
          j_j___libc_free_0(v71, v73 - v71);
        v85 = (const char *)v62;
        v56 = (char *)v60[8];
        if ( v56 == (char *)v60[9] )
          break;
        if ( !v56 )
        {
          v60[8] = 8;
          goto LABEL_94;
        }
        *(_QWORD *)v56 = v62;
        v60[8] += 8;
LABEL_86:
        if ( v68 )
          j_j___libc_free_0(v68, v70 - v68);
        v18 += 152;
        if ( v59 == v18 )
          return sub_187D570(&v65);
      }
      sub_142DF10(v60 + 7, v56, &v85);
      v62 = (__int64)v85;
LABEL_94:
      if ( v62 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v62 + 8LL))(v62);
      goto LABEL_86;
    }
    return sub_187D570(&v65);
  }
}
