// Function: sub_E1E740
// Address: 0xe1e740
//
__int64 __fastcall sub_E1E740(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  char *v6; // rdx
  char *v7; // rax
  char *v8; // rax
  char v9; // al
  __int64 v10; // r8
  __int64 v11; // r9
  char v12; // r14
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _BYTE *v23; // rax
  void *v24; // r14
  __int64 v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r15
  char v36; // al
  __int64 v37; // r13
  __int64 v38; // rax
  char v39; // dl
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rax
  char *v45; // rdx
  __int64 v47; // rdx
  unsigned __int8 *v48; // rax
  __int64 v49; // rcx
  int v50; // r14d
  __int64 v51; // rdx
  __int64 v52; // rax
  char v53; // al
  char v54; // dl
  __int64 v55; // rax
  __int64 v56; // rsi
  int v57; // edx
  char v58; // al
  unsigned __int8 *v59; // rcx
  __int64 v60; // r9
  __int64 v61; // rcx
  __int64 v62; // rax
  __int64 v63; // rax
  char v64; // al
  __int64 v65; // [rsp+8h] [rbp-58h]
  unsigned __int8 v66; // [rsp+8h] [rbp-58h]
  unsigned __int8 v67; // [rsp+8h] [rbp-58h]
  char v68; // [rsp+17h] [rbp-49h]
  __int64 v69; // [rsp+18h] [rbp-48h] BYREF
  __int64 v70[7]; // [rsp+28h] [rbp-38h] BYREF

  v69 = a4;
  v68 = sub_E18460(a1, &v69);
  if ( v68 )
    return 0;
  v6 = (char *)a1[1];
  v7 = (char *)*a1;
  if ( !a3 )
  {
LABEL_32:
    if ( v6 == v7 )
      goto LABEL_9;
    if ( *v7 != 76 )
      goto LABEL_6;
LABEL_34:
    v8 = v7 + 1;
    *a1 = (__int64)v8;
    if ( v8 == v6 )
      goto LABEL_9;
    goto LABEL_7;
  }
  if ( v6 == v7 )
    goto LABEL_9;
  if ( *v7 == 70 )
  {
    ++v7;
    v68 = 1;
    *a1 = (__int64)v7;
    goto LABEL_32;
  }
  if ( *v7 == 76 )
    goto LABEL_34;
LABEL_6:
  v8 = (char *)*a1;
LABEL_7:
  v9 = *v8;
  if ( (unsigned __int8)(v9 - 49) <= 8u )
  {
    v35 = sub_E12DE0(a1);
    goto LABEL_16;
  }
  if ( v9 == 85 )
  {
    v35 = sub_E1E1C0((char **)a1, (__int64)a2);
    goto LABEL_16;
  }
LABEL_9:
  v12 = sub_E0F5E0((const void **)a1, 2u, "DC");
  if ( !v12 )
  {
    v47 = a1[1];
    v48 = (unsigned __int8 *)*a1;
    if ( v47 != *a1 )
    {
      v49 = *v48;
      if ( (unsigned __int8)(v49 - 67) <= 1u )
      {
        if ( !a3 || v69 )
          return 0;
        if ( *(_BYTE *)(a3 + 8) == 48 )
        {
          v55 = sub_E0E790((__int64)(a1 + 102), 16, v47, v49, v10, v11);
          v56 = v55;
          if ( !v55 )
            return 0;
          v11 = 16431;
          v57 = *(_DWORD *)(a3 + 12);
          *(_WORD *)(v55 + 8) = 16431;
          v58 = *(_BYTE *)(v55 + 10);
          *(_DWORD *)(v56 + 12) = v57;
          *(_BYTE *)(v56 + 10) = v58 & 0xF0 | 5;
          *(_QWORD *)v56 = &unk_49DFF68;
          v48 = (unsigned __int8 *)*a1;
          v47 = a1[1];
          if ( *a1 == v47 )
            return 0;
          v49 = *v48;
          a3 = v56;
        }
        if ( (_BYTE)v49 == 67 )
        {
          v59 = v48 + 1;
          *a1 = (__int64)(v48 + 1);
          if ( v48 + 1 == (unsigned __int8 *)v47 )
            return 0;
          v60 = v48[1];
          if ( (_BYTE)v60 == 73 )
          {
            v59 = v48 + 2;
            *a1 = (__int64)(v48 + 2);
            if ( v48 + 2 == (unsigned __int8 *)v47 )
              return 0;
            v60 = v48[2];
            v12 = 1;
          }
          if ( (unsigned __int8)(v60 - 49) > 4u )
            return 0;
          v61 = (__int64)(v59 + 1);
          *a1 = v61;
          if ( a2 )
            *(_BYTE *)a2 = 1;
          if ( v12 )
          {
            v66 = v60;
            v62 = sub_E1D370((__int64)a1, a2, v47, v61, v10, v60);
            v60 = v66;
            if ( !v62 )
              return 0;
          }
          v67 = v60;
          v63 = sub_E0E790((__int64)(a1 + 102), 32, v47, v61, v10, v60);
          v34 = v67;
          v35 = v63;
          if ( !v63 )
            return 0;
          v33 = 16433;
          *(_WORD *)(v63 + 8) = 16433;
          v64 = *(_BYTE *)(v63 + 10);
          *(_QWORD *)(v35 + 16) = a3;
          *(_BYTE *)(v35 + 24) = 0;
          *(_BYTE *)(v35 + 10) = v64 & 0xF0 | 5;
          *(_QWORD *)v35 = &unk_49E0028;
          *(_DWORD *)(v35 + 28) = (char)(v67 - 48);
        }
        else
        {
          if ( (_BYTE)v49 != 68 )
            return 0;
          if ( v47 - (_QWORD)v48 == 1 )
            return 0;
          v50 = v48[1];
          v51 = (unsigned int)(v50 - 48);
          if ( (unsigned __int8)(v50 - 48) > 2u )
          {
            v51 = (unsigned int)(v50 - 52);
            if ( (unsigned __int8)(v50 - 52) > 1u )
              return 0;
          }
          *a1 = (__int64)(v48 + 2);
          if ( a2 )
            *(_BYTE *)a2 = 1;
          v52 = sub_E0E790((__int64)(a1 + 102), 32, v51, v49, v10, v11);
          v35 = v52;
          if ( !v52 )
            return 0;
          *(_WORD *)(v52 + 8) = 16433;
          v53 = *(_BYTE *)(v52 + 10);
          *(_QWORD *)(v35 + 16) = a3;
          *(_BYTE *)(v35 + 24) = 1;
          *(_BYTE *)(v35 + 10) = v53 & 0xF0 | 5;
          *(_QWORD *)v35 = &unk_49E0028;
          *(_DWORD *)(v35 + 28) = (char)v50 - 48;
        }
LABEL_17:
        v37 = v69;
        if ( v69 )
        {
          v38 = sub_E0E790((__int64)(a1 + 102), 32, v31, v32, v33, v34);
          if ( !v38 )
            return 0;
          v39 = *(_BYTE *)(v38 + 10);
          *(_QWORD *)(v38 + 16) = v37;
          *(_WORD *)(v38 + 8) = 16412;
          *(_QWORD *)(v38 + 24) = v35;
          v35 = v38;
          *(_BYTE *)(v38 + 10) = v39 & 0xF0 | 5;
          *(_QWORD *)v38 = &unk_49DF7E8;
        }
        v18 = sub_E0F930(a1, v35);
        if ( v18 )
        {
          if ( v68 )
          {
            v44 = sub_E0E790((__int64)(a1 + 102), 32, v40, v41, v42, v43);
            if ( v44 )
            {
              v54 = *(_BYTE *)(v44 + 10);
              *(_WORD *)(v44 + 8) = 16409;
              *(_BYTE *)(v44 + 10) = v54 & 0xF0 | 5;
              v45 = (char *)&unk_49DF718;
              goto LABEL_25;
            }
          }
          else
          {
            if ( !a3 )
              return v18;
            v44 = sub_E0E790((__int64)(a1 + 102), 32, v40, v41, v42, v43);
            if ( v44 )
            {
              *(_WORD *)(v44 + 8) = 16408;
              *(_BYTE *)(v44 + 10) = *(_BYTE *)(v44 + 10) & 0xF0 | 5;
              v45 = (char *)&unk_49DF6B8;
LABEL_25:
              *(_QWORD *)(v44 + 24) = v18;
              v18 = v44;
              *(_QWORD *)v44 = v45 + 16;
              *(_QWORD *)(v44 + 16) = a3;
              return v18;
            }
          }
        }
        return 0;
      }
    }
    v35 = sub_E1C280((__int64)a1, a2);
LABEL_16:
    if ( !v35 )
      return 0;
    goto LABEL_17;
  }
  v13 = a1[2];
  v65 = a1[3];
  while ( 1 )
  {
    v70[0] = sub_E12DE0(a1);
    v18 = v70[0];
    if ( !v70[0] )
      return v18;
    sub_E18380((__int64)(a1 + 2), v70, v14, v15, v16, v17);
    v23 = (_BYTE *)*a1;
    if ( *a1 != a1[1] && *v23 == 69 )
    {
      *a1 = (__int64)(v23 + 1);
      v24 = sub_E11E80(a1, (v65 - v13) >> 3, v19, v20, v21, v22);
      v26 = v25;
      v30 = sub_E0E790((__int64)(a1 + 102), 32, v25, v27, v28, v29);
      v35 = v30;
      if ( !v30 )
        return 0;
      *(_WORD *)(v30 + 8) = 16437;
      v36 = *(_BYTE *)(v30 + 10);
      *(_QWORD *)(v35 + 16) = v24;
      *(_QWORD *)(v35 + 24) = v26;
      *(_BYTE *)(v35 + 10) = v36 & 0xF0 | 5;
      *(_QWORD *)v35 = &unk_49E01A8;
      goto LABEL_16;
    }
  }
}
