// Function: sub_1BC3F50
// Address: 0x1bc3f50
//
char __fastcall sub_1BC3F50(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v5; // r14
  unsigned __int64 v6; // rdi
  __int64 v7; // r14
  __int64 v8; // rbx
  _QWORD *v9; // r15
  _QWORD *v10; // r13
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 *v21; // rdi
  __int64 v22; // rax
  bool v23; // zf
  __int64 v24; // r14
  int v25; // r8d
  _BYTE *v26; // r9
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rsi
  char *v31; // rdi
  __int64 *v32; // r10
  char *v33; // rsi
  char *v34; // rdi
  char *v35; // rbx
  __int64 v36; // rax
  _QWORD *v37; // rsi
  _BYTE *v38; // rdi
  _QWORD *v39; // r14
  __int64 v40; // rax
  __int64 v42; // [rsp+18h] [rbp-258h]
  __int64 *v43; // [rsp+28h] [rbp-248h]
  __int64 v44; // [rsp+38h] [rbp-238h]
  __int64 v45; // [rsp+38h] [rbp-238h]
  __int64 v46; // [rsp+38h] [rbp-238h]
  __int64 v47; // [rsp+38h] [rbp-238h]
  __int64 v48; // [rsp+38h] [rbp-238h]
  __int64 v49; // [rsp+38h] [rbp-238h]
  __int64 v50; // [rsp+38h] [rbp-238h]
  __int64 v51; // [rsp+38h] [rbp-238h]
  __int64 v52; // [rsp+38h] [rbp-238h]
  __int64 v53; // [rsp+38h] [rbp-238h]
  __int64 v54; // [rsp+40h] [rbp-230h]
  __int64 v55; // [rsp+50h] [rbp-220h] BYREF
  int v56; // [rsp+58h] [rbp-218h] BYREF
  _BYTE v57[16]; // [rsp+60h] [rbp-210h] BYREF
  __int64 v58; // [rsp+70h] [rbp-200h]
  char v59; // [rsp+80h] [rbp-1F0h]
  _BYTE *v60; // [rsp+90h] [rbp-1E0h]
  __int64 v61; // [rsp+98h] [rbp-1D8h]
  _BYTE v62[192]; // [rsp+A0h] [rbp-1D0h] BYREF
  __int64 v63; // [rsp+160h] [rbp-110h] BYREF
  char *v64; // [rsp+168h] [rbp-108h] BYREF
  __int64 v65; // [rsp+170h] [rbp-100h]
  _BYTE v66[248]; // [rsp+178h] [rbp-F8h] BYREF

  v54 = (__int64)(a1 + 9);
  sub_196A810((__int64)(a1 + 9));
  v3 = a1[13];
  v4 = a1[14];
  v43 = a1 + 13;
  if ( v3 != v4 )
  {
    v5 = a1[13];
    do
    {
      v6 = *(_QWORD *)(v5 + 8);
      if ( v6 != v5 + 24 )
        _libc_free(v6);
      v5 += 88;
    }
    while ( v4 != v5 );
    a1[14] = v3;
  }
  sub_196A810((__int64)(a1 + 16));
  v7 = a1[21];
  v44 = a1[20];
  if ( v44 != v7 )
  {
    v8 = a1[20];
    do
    {
      v9 = *(_QWORD **)(v8 + 8);
      v10 = &v9[3 * *(unsigned int *)(v8 + 16)];
      if ( v9 != v10 )
      {
        do
        {
          v11 = *(v10 - 1);
          v10 -= 3;
          if ( v11 != -8 && v11 != 0 && v11 != -16 )
            sub_1649B30(v10);
        }
        while ( v9 != v10 );
        v10 = *(_QWORD **)(v8 + 8);
      }
      if ( v10 != (_QWORD *)(v8 + 24) )
        _libc_free((unsigned __int64)v10);
      v8 += 216;
    }
    while ( v7 != v8 );
    a1[21] = v44;
  }
  v12 = *(_QWORD *)(a2 + 48);
  v13 = a2 + 40;
  v14 = (__int64)v57;
  if ( v12 != a2 + 40 )
  {
    while ( 1 )
    {
      if ( !v12 )
        BUG();
      LOBYTE(v14) = *(_BYTE *)(v12 - 8);
      v16 = v12 - 24;
      if ( (_BYTE)v14 == 55 )
        break;
      if ( (_BYTE)v14 != 56 )
        goto LABEL_25;
      v14 = *(_DWORD *)(v12 - 4) & 0xFFFFFFF;
      if ( (unsigned int)(v14 - 1) > 1 )
        goto LABEL_25;
      v14 = *(_QWORD *)(v16 + 24 * (1 - v14));
      if ( *(_BYTE *)(v14 + 16) <= 0x10u )
        goto LABEL_25;
      v45 = *(_QWORD *)v14;
      LOBYTE(v14) = sub_1643F10(*(_QWORD *)v14);
      if ( !(_BYTE)v14 )
        goto LABEL_25;
      LOBYTE(v14) = *(_BYTE *)(v45 + 8) & 0xFD;
      if ( (_BYTE)v14 == 4 )
        goto LABEL_25;
      v14 = *(_QWORD *)(v12 - 24);
      if ( *(_BYTE *)(v14 + 8) == 16 )
        goto LABEL_25;
      v55 = sub_14AD280(*(_QWORD *)(v16 - 24LL * (*(_DWORD *)(v12 - 4) & 0xFFFFFFF)), a1[8], 6u);
      v42 = v55;
      v56 = 0;
      sub_1BC3CA0((__int64)v57, (__int64)(a1 + 16), &v55, &v56);
      v46 = v58;
      if ( v59 )
      {
        v63 = v42;
        v32 = (__int64 *)a1[21];
        v60 = v62;
        v61 = 0x800000000LL;
        v64 = v66;
        v65 = 0x800000000LL;
        if ( v32 == (__int64 *)a1[22] )
        {
          sub_1BC3640(a1 + 20, v32, (__int64)&v63);
          v33 = v64;
          v34 = &v64[24 * (unsigned int)v65];
        }
        else
        {
          if ( v32 )
          {
            *v32 = v42;
            v32[1] = (__int64)(v32 + 3);
            v32[2] = 0x800000000LL;
            if ( (_DWORD)v65 )
            {
              sub_1BC1780((__int64)(v32 + 1), (__int64)&v64);
              v33 = v64;
              v32 = (__int64 *)a1[21];
              v34 = &v64[24 * (unsigned int)v65];
            }
            else
            {
              v33 = v64;
              v32 = (__int64 *)a1[21];
              v34 = v64;
            }
          }
          else
          {
            v34 = v66;
            v33 = v66;
          }
          a1[21] = v32 + 27;
        }
        if ( v33 != v34 )
        {
          v35 = v34;
          do
          {
            v36 = *((_QWORD *)v35 - 1);
            v35 -= 24;
            if ( v36 != -8 && v36 != 0 && v36 != -16 )
              sub_1649B30(v35);
          }
          while ( v35 != v33 );
          v16 = v12 - 24;
          v34 = v64;
        }
        if ( v34 != v66 )
          _libc_free((unsigned __int64)v34);
        v37 = v60;
        v38 = &v60[24 * (unsigned int)v61];
        if ( v60 != v38 )
        {
          v39 = &v60[24 * (unsigned int)v61];
          do
          {
            v40 = *(v39 - 1);
            v39 -= 3;
            if ( v40 != 0 && v40 != -8 && v40 != -16 )
              sub_1649B30(v39);
          }
          while ( v37 != v39 );
          v38 = v60;
        }
        if ( v38 != v62 )
          _libc_free((unsigned __int64)v38);
        v17 = 1749801491 * (unsigned int)((__int64)(a1[21] - a1[20]) >> 3) - 1;
        *(_DWORD *)(v46 + 8) = v17;
      }
      else
      {
        v17 = *(unsigned int *)(v58 + 8);
      }
      v65 = v16;
      v63 = 6;
      v18 = 27 * v17;
      v19 = a1[20];
      v64 = 0;
      v20 = v19 + 8 * v18;
      if ( v16 != -8 && v16 != -16 )
      {
        v47 = v20;
        sub_164C220((__int64)&v63);
        v20 = v47;
      }
      LODWORD(v14) = *(_DWORD *)(v20 + 16);
      if ( (unsigned int)v14 >= *(_DWORD *)(v20 + 20) )
      {
        v53 = v20;
        sub_170B450(v20 + 8, 0);
        v20 = v53;
        LODWORD(v14) = *(_DWORD *)(v53 + 16);
      }
      v21 = (unsigned __int64 *)(*(_QWORD *)(v20 + 8) + 24LL * (unsigned int)v14);
      if ( v21 )
      {
        *v21 = 6;
        v21[1] = 0;
        v22 = v65;
        v23 = v65 == 0;
        v21[2] = v65;
        if ( v22 != -8 && !v23 && v22 != -16 )
        {
          v48 = v20;
          sub_1649AC0(v21, v63 & 0xFFFFFFFFFFFFFFF8LL);
          v20 = v48;
        }
        LODWORD(v14) = *(_DWORD *)(v20 + 16);
      }
      *(_DWORD *)(v20 + 16) = v14 + 1;
      LOBYTE(v14) = v65;
      if ( v65 == -8 || v65 == 0 || v65 == -16 )
      {
LABEL_25:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          return v14;
      }
      else
      {
        LOBYTE(v14) = sub_1649B30(&v63);
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          return v14;
      }
    }
    LOBYTE(v14) = sub_15F32D0(v12 - 24);
    if ( !(_BYTE)v14 && (*(_BYTE *)(v12 - 6) & 1) == 0 )
    {
      v15 = **(_QWORD **)(v12 - 72);
      LOBYTE(v14) = sub_1643F10(v15);
      if ( (_BYTE)v14 )
      {
        LOBYTE(v14) = *(_BYTE *)(v15 + 8) & 0xFD;
        if ( (_BYTE)v14 != 4 )
        {
          v55 = sub_14AD280(*(_QWORD *)(v12 - 48), a1[8], 6u);
          v24 = v55;
          v56 = 0;
          sub_1BC3CA0((__int64)v57, v54, &v55, &v56);
          v27 = v58;
          if ( v59 )
          {
            v26 = v66;
            v25 = 0;
            v63 = v24;
            v60 = v62;
            v30 = a1[14];
            v61 = 0x800000000LL;
            v64 = v66;
            v65 = 0x800000000LL;
            if ( v30 == a1[15] )
            {
              v52 = v58;
              sub_196D260(v43, (char *)v30, (__int64)&v63, v58);
              v31 = v64;
              v26 = v66;
              v27 = v52;
            }
            else
            {
              v31 = v66;
              if ( v30 )
              {
                *(_QWORD *)v30 = v24;
                *(_QWORD *)(v30 + 8) = v30 + 24;
                *(_QWORD *)(v30 + 16) = 0x800000000LL;
                if ( (_DWORD)v65 )
                {
                  v50 = v27;
                  sub_1BB96B0(v30 + 8, &v64, (unsigned int)v65, v27, 0, (int)v66);
                  v30 = a1[14];
                  v31 = v64;
                  v26 = v66;
                  v27 = v50;
                }
                else
                {
                  v30 = a1[14];
                  v31 = v64;
                }
              }
              a1[14] = v30 + 88;
            }
            if ( v31 != v66 )
            {
              v49 = v27;
              _libc_free((unsigned __int64)v31);
              v27 = v49;
            }
            v28 = -1171354717 * (unsigned int)((__int64)(a1[14] - a1[13]) >> 3) - 1;
            *(_DWORD *)(v27 + 8) = v28;
          }
          else
          {
            v28 = *(unsigned int *)(v58 + 8);
          }
          v14 = a1[13] + 88 * v28;
          v29 = *(unsigned int *)(v14 + 16);
          if ( (unsigned int)v29 >= *(_DWORD *)(v14 + 20) )
          {
            v51 = v14;
            sub_16CD150(v14 + 8, (const void *)(v14 + 24), 0, 8, v25, (int)v26);
            v14 = v51;
            v29 = *(unsigned int *)(v51 + 16);
          }
          *(_QWORD *)(*(_QWORD *)(v14 + 8) + 8 * v29) = v16;
          ++*(_DWORD *)(v14 + 16);
        }
      }
    }
    goto LABEL_25;
  }
  return v14;
}
