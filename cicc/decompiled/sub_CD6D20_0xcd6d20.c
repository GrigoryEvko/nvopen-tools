// Function: sub_CD6D20
// Address: 0xcd6d20
//
void __fastcall sub_CD6D20(__int64 a1, unsigned int *a2)
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
  __int64 v37; // rsi
  __int64 *v38; // rdi
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  int v41; // ecx
  __int64 v42; // rdx
  _BYTE *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 *v46; // rdi
  __int64 v47; // rsi
  __int64 v48; // rax
  int v49; // ecx
  __int64 v50; // rdx
  _BYTE *v51; // rdi
  __int64 v52; // r8
  __int64 v53; // rax
  __int64 v54; // rsi
  __int64 *v55; // rdi
  __int64 v56; // rdx
  unsigned __int64 v57; // rax
  int v58; // ecx
  __int64 v59; // rdx
  unsigned int v60; // [rsp+Ch] [rbp-84h]
  unsigned int v61; // [rsp+Ch] [rbp-84h]
  char v62; // [rsp+1Fh] [rbp-71h] BYREF
  _BYTE *v63; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v64; // [rsp+28h] [rbp-68h]
  _BYTE *v65; // [rsp+30h] [rbp-60h]
  __int64 v66; // [rsp+40h] [rbp-50h] BYREF
  __int64 v67; // [rsp+48h] [rbp-48h]
  __int64 v68; // [rsp+50h] [rbp-40h]

  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *a2 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, _BYTE **, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "NumViews",
         0,
         v5,
         &v63,
         &v66) )
  {
    sub_CCC2C0(a1, a2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
  }
  else if ( (_BYTE)v63 )
  {
    *a2 = 0;
  }
  LODWORD(v63) = a2[4] & 1;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = (_DWORD)v63 == 0;
  v8 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "IsImplicit",
         0,
         v7,
         &v62,
         &v66);
  if ( v8 )
  {
    sub_CCC2C0(a1, (unsigned int *)&v63);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
    v8 = (unsigned __int8)v63 & 1;
  }
  else if ( !v62 )
  {
    v8 = (unsigned __int8)v63 & 1;
  }
  v9 = v8 & 1 | a2[4] & 0xFE;
  *((_BYTE *)a2 + 16) = v9;
  LODWORD(v63) = (v9 & 2) != 0;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = (_DWORD)v63 == 0;
  v12 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "ComputePerPatchAttribsForViewZero",
          0,
          v11,
          &v62,
          &v66);
  if ( v12 )
  {
    sub_CCC2C0(a1, (unsigned int *)&v63);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
    v12 = (unsigned __int8)v63 & 1;
  }
  else if ( !v62 )
  {
    v12 = (unsigned __int8)v63 & 1;
  }
  v13 = (2 * (v12 & 1)) | a2[4] & 0xFD;
  *((_BYTE *)a2 + 16) = v13;
  LODWORD(v63) = (v13 & 4) != 0;
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v15 = 0;
  if ( v14 )
    v15 = (_DWORD)v63 == 0;
  v16 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "EnableViewInstanceMask",
          0,
          v15,
          &v62,
          &v66);
  if ( v16 )
  {
    sub_CCC2C0(a1, (unsigned int *)&v63);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
    v16 = (unsigned __int8)v63 & 1;
  }
  else if ( !v62 )
  {
    v16 = (unsigned __int8)v63 & 1;
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
         &v63,
         &v66) )
  {
    sub_CCC2C0(a1, a2 + 5);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
  }
  else if ( (_BYTE)v63 )
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
         &v63,
         &v66) )
  {
    sub_CCC2C0(a1, a2 + 6);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v66);
  }
  else if ( (_BYTE)v63 )
  {
    a2[6] = 0;
  }
  v21 = sub_CB0A70(a1);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    sub_CD6C30(a1, (__int64)"NominalViewIDs", (__int64)&v63, (__int64)&v66, 0);
    if ( v66 )
      j_j___libc_free_0(v66, v68 - v66);
    v43 = v63;
    if ( v63 == v64 )
    {
      *((_QWORD *)a2 + 1) = 0;
      if ( !v43 )
        goto LABEL_44;
    }
    else
    {
      v44 = sub_2207820((v64 - v63) >> 2);
      v45 = v64 - v63;
      *((_QWORD *)a2 + 1) = v44;
      v46 = *(__int64 **)(v21 + 8);
      v47 = v45 >> 2;
      v48 = *v46;
      v46[10] += v47;
      if ( v46[1] >= (unsigned __int64)(v48 + v47) && v48 )
        *v46 = v48 + v47;
      else
        v48 = sub_9D1E70((__int64)v46, v47, v47, 0);
      *((_QWORD *)a2 + 1) = v48;
      v43 = v63;
      if ( v64 != v63 )
      {
        v49 = 0;
        v50 = 0;
        while ( 1 )
        {
          *(_BYTE *)(v48 + v50) = *(_DWORD *)&v43[4 * v50];
          v43 = v63;
          v50 = (unsigned int)++v49;
          if ( v49 == (v64 - v63) >> 2 )
            break;
          v48 = *((_QWORD *)a2 + 1);
        }
      }
      if ( !v43 )
        goto LABEL_44;
    }
    j_j___libc_free_0(v43, v65 - v43);
    goto LABEL_44;
  }
  if ( *((_QWORD *)a2 + 1) )
  {
    v22 = *a2;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    sub_C17980((__int64)&v63, v22);
    v23 = 0;
    if ( *a2 )
    {
      do
      {
        while ( 1 )
        {
          v24 = v64;
          v25 = *(unsigned __int8 *)(*((_QWORD *)a2 + 1) + v23);
          LODWORD(v66) = v25;
          if ( v64 != v65 )
            break;
          v60 = v23;
          sub_C88AB0((__int64)&v63, v64, &v66);
          v23 = v60 + 1;
          if ( *a2 == v60 + 1 )
            goto LABEL_40;
        }
        if ( v64 )
        {
          *(_DWORD *)v64 = v25;
          v24 = v64;
        }
        ++v23;
        v64 = v24 + 4;
      }
      while ( *a2 != v23 );
    }
LABEL_40:
    v66 = 0;
    v67 = 0;
    v68 = 0;
    sub_CD6C30(a1, (__int64)"NominalViewIDs", (__int64)&v63, (__int64)&v66, 0);
    if ( v66 )
      j_j___libc_free_0(v66, v68 - v66);
    if ( v63 )
      j_j___libc_free_0(v63, v65 - v63);
  }
LABEL_44:
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( *((_QWORD *)a2 + 4) )
    {
      v26 = *a2;
      v63 = 0;
      v64 = 0;
      v65 = 0;
      sub_C17980((__int64)&v63, v26);
      v27 = 0;
      if ( *a2 )
      {
        do
        {
          while ( 1 )
          {
            v28 = v64;
            v29 = *(_DWORD *)(*((_QWORD *)a2 + 4) + 4LL * v27);
            LODWORD(v66) = v29;
            if ( v64 != v65 )
              break;
            v61 = v27;
            sub_C88AB0((__int64)&v63, v64, &v66);
            v27 = v61 + 1;
            if ( *a2 == v61 + 1 )
              goto LABEL_53;
          }
          if ( v64 )
          {
            *(_DWORD *)v64 = v29;
            v28 = v64;
          }
          ++v27;
          v64 = v28 + 4;
        }
        while ( *a2 != v27 );
      }
LABEL_53:
      v66 = 0;
      v67 = 0;
      v68 = 0;
      sub_CD6C30(a1, (__int64)"PerViewRTIndexConstants", (__int64)&v63, (__int64)&v66, 0);
      if ( v66 )
        j_j___libc_free_0(v66, v68 - v66);
      if ( v63 )
        j_j___libc_free_0(v63, v65 - v63);
    }
  }
  else
  {
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    sub_CD6C30(a1, (__int64)"PerViewRTIndexConstants", (__int64)&v63, (__int64)&v66, 0);
    if ( v66 )
      j_j___libc_free_0(v66, v68 - v66);
    v51 = v63;
    if ( v64 == v63 )
    {
      *((_QWORD *)a2 + 4) = 0;
    }
    else
    {
      v52 = v64 - v63;
      if ( (unsigned __int64)(v64 - v63) > 0x7FFFFFFFFFFFFFF8LL )
        v52 = -1;
      v53 = sub_2207820(v52);
      v54 = v64 - v63;
      *((_QWORD *)a2 + 4) = v53;
      v55 = *(__int64 **)(v21 + 8);
      v56 = *v55;
      v55[10] += v54;
      v57 = (v56 + 3) & 0xFFFFFFFFFFFFFFFCLL;
      if ( v55[1] >= v54 + v57 && v56 )
        *v55 = v54 + v57;
      else
        v57 = sub_9D1E70((__int64)v55, v54, v54, 2);
      *((_QWORD *)a2 + 4) = v57;
      v51 = v63;
      if ( v64 != v63 )
      {
        v58 = 0;
        v59 = 0;
        while ( 1 )
        {
          *(_DWORD *)(v57 + 4 * v59) = *(_DWORD *)&v51[4 * v59];
          v51 = v63;
          v59 = (unsigned int)++v58;
          if ( v58 == (v64 - v63) >> 2 )
            break;
          v57 = *((_QWORD *)a2 + 4);
        }
      }
    }
    if ( v51 )
      j_j___libc_free_0(v51, v65 - v51);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( *((_QWORD *)a2 + 5) )
    {
      v30 = *a2;
      v63 = 0;
      v31 = 0;
      v64 = 0;
      v65 = 0;
      sub_C17980((__int64)&v63, v30);
      if ( *a2 )
      {
        do
        {
          while ( 1 )
          {
            v32 = v64;
            v33 = *(_DWORD *)(*((_QWORD *)a2 + 5) + 4LL * v31);
            LODWORD(v66) = v33;
            if ( v64 != v65 )
              break;
            ++v31;
            sub_C88AB0((__int64)&v63, v64, &v66);
            if ( *a2 == v31 )
              goto LABEL_66;
          }
          if ( v64 )
          {
            *(_DWORD *)v64 = v33;
            v32 = v64;
          }
          ++v31;
          v64 = v32 + 4;
        }
        while ( *a2 != v31 );
      }
LABEL_66:
      v66 = 0;
      v67 = 0;
      v68 = 0;
      sub_CD6C30(a1, (__int64)"PerViewVPIndexConstants", (__int64)&v63, (__int64)&v66, 0);
      if ( v66 )
        j_j___libc_free_0(v66, v68 - v66);
      if ( v63 )
        j_j___libc_free_0(v63, v65 - v63);
    }
  }
  else
  {
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v67 = 0;
    v68 = 0;
    sub_CD6C30(a1, (__int64)"PerViewVPIndexConstants", (__int64)&v63, (__int64)&v66, 0);
    if ( v66 )
      j_j___libc_free_0(v66, v68 - v66);
    v34 = v63;
    if ( v64 == v63 )
    {
      *((_QWORD *)a2 + 5) = 0;
    }
    else
    {
      v35 = v64 - v63;
      if ( (unsigned __int64)(v64 - v63) > 0x7FFFFFFFFFFFFFF8LL )
        v35 = -1;
      v36 = sub_2207820(v35);
      v37 = v64 - v63;
      *((_QWORD *)a2 + 5) = v36;
      v38 = *(__int64 **)(v21 + 8);
      v39 = *v38;
      v38[10] += v37;
      v40 = (v39 + 3) & 0xFFFFFFFFFFFFFFFCLL;
      if ( v38[1] >= v37 + v40 && v39 )
        *v38 = v37 + v40;
      else
        v40 = sub_9D1E70((__int64)v38, v37, v37, 2);
      *((_QWORD *)a2 + 5) = v40;
      v34 = v63;
      if ( v64 != v63 )
      {
        v41 = 0;
        v42 = 0;
        while ( 1 )
        {
          *(_DWORD *)(v40 + 4 * v42) = *(_DWORD *)&v34[4 * v42];
          v34 = v63;
          v42 = (unsigned int)++v41;
          if ( v41 == (v64 - v63) >> 2 )
            break;
          v40 = *((_QWORD *)a2 + 5);
        }
      }
    }
    if ( v34 )
      j_j___libc_free_0(v34, v65 - v34);
  }
}
