// Function: sub_1C1D5F0
// Address: 0x1c1d5f0
//
void __fastcall sub_1C1D5F0(__int64 a1, unsigned int *a2)
{
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  char v9; // al
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  char v13; // al
  char v14; // al
  _BOOL8 v15; // rcx
  char v16; // al
  char v17; // al
  _BOOL8 v18; // rcx
  char v19; // al
  _BOOL8 v20; // rcx
  __int64 v21; // r15
  unsigned __int64 v22; // rsi
  unsigned int v23; // eax
  _BYTE *v24; // rsi
  int v25; // edx
  unsigned __int64 v26; // rsi
  unsigned int v27; // eax
  _BYTE *v28; // rsi
  int v29; // edx
  unsigned __int64 v30; // rsi
  unsigned int v31; // r15d
  _BYTE *v32; // rsi
  int v33; // eax
  _BYTE *v34; // rdi
  __int64 v35; // r8
  __int64 v36; // rax
  _BYTE *v37; // rsi
  __int64 v38; // rax
  int v39; // ecx
  __int64 v40; // rdx
  _BYTE *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rax
  int v45; // ecx
  __int64 v46; // rdx
  _BYTE *v47; // rdi
  __int64 v48; // r8
  __int64 v49; // rax
  _BYTE *v50; // rsi
  __int64 v51; // rax
  int v52; // ecx
  __int64 v53; // rdx
  unsigned int v54; // [rsp+Ch] [rbp-84h]
  unsigned int v55; // [rsp+Ch] [rbp-84h]
  char v56; // [rsp+1Fh] [rbp-71h] BYREF
  _BYTE *v57; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v58; // [rsp+28h] [rbp-68h]
  _BYTE *v59; // [rsp+30h] [rbp-60h]
  __int64 v60; // [rsp+40h] [rbp-50h] BYREF
  __int64 v61; // [rsp+48h] [rbp-48h]
  __int64 v62; // [rsp+50h] [rbp-40h]

  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *a2 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NumViews",
         0,
         v5,
         &v57,
         &v60) )
  {
    sub_1C14710(a1, a2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
  }
  else if ( (_BYTE)v57 )
  {
    *a2 = 0;
  }
  LODWORD(v57) = a2[4] & 1;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = (_DWORD)v57 == 0;
  v8 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IsImplicit",
         0,
         v7,
         &v56,
         &v60);
  if ( v8 )
  {
    sub_1C14710(a1, (unsigned int *)&v57);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
    v8 = (unsigned __int8)v57 & 1;
  }
  else if ( !v56 )
  {
    v8 = (unsigned __int8)v57 & 1;
  }
  v9 = v8 & 1 | a2[4] & 0xFE;
  *((_BYTE *)a2 + 16) = v9;
  LODWORD(v57) = (v9 & 2) != 0;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = (_DWORD)v57 == 0;
  v12 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "ComputePerPatchAttribsForViewZero",
          0,
          v11,
          &v56,
          &v60);
  if ( v12 )
  {
    sub_1C14710(a1, (unsigned int *)&v57);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
    v12 = (unsigned __int8)v57 & 1;
  }
  else if ( !v56 )
  {
    v12 = (unsigned __int8)v57 & 1;
  }
  v13 = (2 * (v12 & 1)) | a2[4] & 0xFD;
  *((_BYTE *)a2 + 16) = v13;
  LODWORD(v57) = (v13 & 4) != 0;
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v15 = 0;
  if ( v14 )
    v15 = (_DWORD)v57 == 0;
  v16 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "EnableViewInstanceMask",
          0,
          v15,
          &v56,
          &v60);
  if ( v16 )
  {
    sub_1C14710(a1, (unsigned int *)&v57);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
    v16 = (unsigned __int8)v57 & 1;
  }
  else if ( !v56 )
  {
    v16 = (unsigned __int8)v57 & 1;
  }
  *((_BYTE *)a2 + 16) = a2[4] & 0xFB | (4 * (v16 & 1));
  v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v18 = 0;
  if ( v17 )
    v18 = a2[5] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ViewInstanceMaskBank",
         0,
         v18,
         &v57,
         &v60) )
  {
    sub_1C14710(a1, a2 + 5);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
  }
  else if ( (_BYTE)v57 )
  {
    a2[5] = 0;
  }
  v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v20 = 0;
  if ( v19 )
    v20 = a2[6] == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ViewInstanceMaskByteOffset",
         0,
         v20,
         &v57,
         &v60) )
  {
    sub_1C14710(a1, a2 + 6);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v60);
  }
  else if ( (_BYTE)v57 )
  {
    a2[6] = 0;
  }
  v21 = sub_16E4080(a1);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    sub_1C1D500(a1, (__int64)"NominalViewIDs", (__int64)&v57, (__int64)&v60, 0);
    if ( v60 )
      j_j___libc_free_0(v60, v62 - v60);
    v41 = v57;
    if ( v57 == v58 )
    {
      *((_QWORD *)a2 + 1) = 0;
      if ( !v41 )
        goto LABEL_44;
    }
    else
    {
      v42 = sub_2207820((v58 - v57) >> 2);
      v43 = v58 - v57;
      *((_QWORD *)a2 + 1) = v42;
      v44 = sub_145CBF0(*(__int64 **)(v21 + 8), v43 >> 2, 1);
      v41 = v57;
      *((_QWORD *)a2 + 1) = v44;
      if ( v41 != v58 )
      {
        v45 = 0;
        v46 = 0;
        while ( 1 )
        {
          *(_BYTE *)(v44 + v46) = *(_DWORD *)&v41[4 * v46];
          v41 = v57;
          v46 = (unsigned int)++v45;
          if ( v45 == (v58 - v57) >> 2 )
            break;
          v44 = *((_QWORD *)a2 + 1);
        }
      }
      if ( !v41 )
        goto LABEL_44;
    }
    j_j___libc_free_0(v41, v59 - v41);
    goto LABEL_44;
  }
  if ( *((_QWORD *)a2 + 1) )
  {
    v22 = *a2;
    v57 = 0;
    v58 = 0;
    v59 = 0;
    sub_C17980((__int64)&v57, v22);
    v23 = 0;
    if ( *a2 )
    {
      do
      {
        while ( 1 )
        {
          v24 = v58;
          v25 = *(unsigned __int8 *)(*((_QWORD *)a2 + 1) + v23);
          LODWORD(v60) = v25;
          if ( v58 != v59 )
            break;
          v54 = v23;
          sub_C88AB0((__int64)&v57, v58, &v60);
          v23 = v54 + 1;
          if ( *a2 == v54 + 1 )
            goto LABEL_40;
        }
        if ( v58 )
        {
          *(_DWORD *)v58 = v25;
          v24 = v58;
        }
        ++v23;
        v58 = v24 + 4;
      }
      while ( *a2 != v23 );
    }
LABEL_40:
    v60 = 0;
    v61 = 0;
    v62 = 0;
    sub_1C1D500(a1, (__int64)"NominalViewIDs", (__int64)&v57, (__int64)&v60, 0);
    if ( v60 )
      j_j___libc_free_0(v60, v62 - v60);
    if ( v57 )
      j_j___libc_free_0(v57, v59 - v57);
  }
LABEL_44:
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( *((_QWORD *)a2 + 4) )
    {
      v26 = *a2;
      v57 = 0;
      v58 = 0;
      v59 = 0;
      sub_C17980((__int64)&v57, v26);
      v27 = 0;
      if ( *a2 )
      {
        do
        {
          while ( 1 )
          {
            v28 = v58;
            v29 = *(_DWORD *)(*((_QWORD *)a2 + 4) + 4LL * v27);
            LODWORD(v60) = v29;
            if ( v58 != v59 )
              break;
            v55 = v27;
            sub_C88AB0((__int64)&v57, v58, &v60);
            v27 = v55 + 1;
            if ( *a2 == v55 + 1 )
              goto LABEL_53;
          }
          if ( v58 )
          {
            *(_DWORD *)v58 = v29;
            v28 = v58;
          }
          ++v27;
          v58 = v28 + 4;
        }
        while ( *a2 != v27 );
      }
LABEL_53:
      v60 = 0;
      v61 = 0;
      v62 = 0;
      sub_1C1D500(a1, (__int64)"PerViewRTIndexConstants", (__int64)&v57, (__int64)&v60, 0);
      if ( v60 )
        j_j___libc_free_0(v60, v62 - v60);
      if ( v57 )
        j_j___libc_free_0(v57, v59 - v57);
    }
  }
  else
  {
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    sub_1C1D500(a1, (__int64)"PerViewRTIndexConstants", (__int64)&v57, (__int64)&v60, 0);
    if ( v60 )
      j_j___libc_free_0(v60, v62 - v60);
    v47 = v57;
    if ( v58 == v57 )
    {
      *((_QWORD *)a2 + 4) = 0;
    }
    else
    {
      v48 = v58 - v57;
      if ( (unsigned __int64)(v58 - v57) > 0x7FFFFFFFFFFFFFF8LL )
        v48 = -1;
      v49 = sub_2207820(v48);
      v50 = (_BYTE *)(v58 - v57);
      *((_QWORD *)a2 + 4) = v49;
      v51 = sub_145CBF0(*(__int64 **)(v21 + 8), (__int64)v50, 4);
      v47 = v57;
      *((_QWORD *)a2 + 4) = v51;
      if ( v47 != v58 )
      {
        v52 = 0;
        v53 = 0;
        while ( 1 )
        {
          *(_DWORD *)(v51 + 4 * v53) = *(_DWORD *)&v47[4 * v53];
          v47 = v57;
          v53 = (unsigned int)++v52;
          if ( v52 == (v58 - v57) >> 2 )
            break;
          v51 = *((_QWORD *)a2 + 4);
        }
      }
    }
    if ( v47 )
      j_j___libc_free_0(v47, v59 - v47);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( *((_QWORD *)a2 + 5) )
    {
      v30 = *a2;
      v57 = 0;
      v31 = 0;
      v58 = 0;
      v59 = 0;
      sub_C17980((__int64)&v57, v30);
      if ( *a2 )
      {
        do
        {
          while ( 1 )
          {
            v32 = v58;
            v33 = *(_DWORD *)(*((_QWORD *)a2 + 5) + 4LL * v31);
            LODWORD(v60) = v33;
            if ( v58 != v59 )
              break;
            ++v31;
            sub_C88AB0((__int64)&v57, v58, &v60);
            if ( *a2 == v31 )
              goto LABEL_66;
          }
          if ( v58 )
          {
            *(_DWORD *)v58 = v33;
            v32 = v58;
          }
          ++v31;
          v58 = v32 + 4;
        }
        while ( *a2 != v31 );
      }
LABEL_66:
      v60 = 0;
      v61 = 0;
      v62 = 0;
      sub_1C1D500(a1, (__int64)"PerViewVPIndexConstants", (__int64)&v57, (__int64)&v60, 0);
      if ( v60 )
        j_j___libc_free_0(v60, v62 - v60);
      if ( v57 )
        j_j___libc_free_0(v57, v59 - v57);
    }
  }
  else
  {
    v57 = 0;
    v58 = 0;
    v59 = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    sub_1C1D500(a1, (__int64)"PerViewVPIndexConstants", (__int64)&v57, (__int64)&v60, 0);
    if ( v60 )
      j_j___libc_free_0(v60, v62 - v60);
    v34 = v57;
    if ( v58 == v57 )
    {
      *((_QWORD *)a2 + 5) = 0;
    }
    else
    {
      v35 = v58 - v57;
      if ( (unsigned __int64)(v58 - v57) > 0x7FFFFFFFFFFFFFF8LL )
        v35 = -1;
      v36 = sub_2207820(v35);
      v37 = (_BYTE *)(v58 - v57);
      *((_QWORD *)a2 + 5) = v36;
      v38 = sub_145CBF0(*(__int64 **)(v21 + 8), (__int64)v37, 4);
      v34 = v57;
      *((_QWORD *)a2 + 5) = v38;
      if ( v34 != v58 )
      {
        v39 = 0;
        v40 = 0;
        while ( 1 )
        {
          *(_DWORD *)(v38 + 4 * v40) = *(_DWORD *)&v34[4 * v40];
          v34 = v57;
          v40 = (unsigned int)++v39;
          if ( v39 == (v58 - v57) >> 2 )
            break;
          v38 = *((_QWORD *)a2 + 5);
        }
      }
    }
    if ( v34 )
      j_j___libc_free_0(v34, v59 - v34);
  }
}
