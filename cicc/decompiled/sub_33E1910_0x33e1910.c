// Function: sub_33E1910
// Address: 0x33e1910
//
__int64 __fastcall sub_33E1910(unsigned int a1, char a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // r13d
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  unsigned __int16 *v16; // r12
  int v17; // eax
  _QWORD *v18; // r12
  char **v19; // r14
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  void *v23; // rcx
  void *v24; // rax
  void *v25; // rdx
  unsigned int v26; // r9d
  int v28; // ebx
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  _QWORD *v32; // rax
  void *v33; // r12
  void *v34; // rax
  void *v35; // r13
  __int64 v36; // r8
  __int64 v37; // rbx
  __int64 v38; // rbx
  _QWORD *v39; // rdx
  unsigned int v40; // ebx
  unsigned int v41; // ebx
  int v42; // ebx
  __int64 v43; // r14
  _DWORD *v44; // r15
  __int64 v45; // r13
  unsigned __int64 v46; // r15
  void *v47; // rax
  void *v48; // rbx
  __int64 v49; // rbx
  __int64 v50; // rbx
  __int64 v51; // rbx
  _DWORD *v52; // r14
  void *v53; // r13
  bool v54; // r9
  __int64 v55; // rdi
  __int64 v56; // rdi
  unsigned int v57; // eax
  _QWORD *i; // rbx
  __int64 v59; // [rsp+8h] [rbp-88h]
  bool v60; // [rsp+8h] [rbp-88h]
  unsigned __int8 v61; // [rsp+8h] [rbp-88h]
  unsigned __int8 v62; // [rsp+8h] [rbp-88h]
  bool v63; // [rsp+8h] [rbp-88h]
  __int64 v64; // [rsp+8h] [rbp-88h]
  char v65; // [rsp+8h] [rbp-88h]
  unsigned __int8 v66; // [rsp+8h] [rbp-88h]
  __int16 v67; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v68; // [rsp+18h] [rbp-78h]
  unsigned __int64 v69; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v70; // [rsp+28h] [rbp-68h]
  void *v71[10]; // [rsp+40h] [rbp-50h] BYREF

  v8 = a5;
  v11 = sub_33DFBC0(a3, a4, 0, 1u, a5, a6);
  if ( v11 )
  {
    v15 = *(_QWORD *)(v11 + 96);
    v16 = (unsigned __int16 *)(*(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4);
    v17 = *v16;
    v18 = (_QWORD *)*((_QWORD *)v16 + 1);
    v19 = (char **)(v15 + 24);
    v67 = v17;
    v68 = v18;
    if ( (_WORD)v17 )
    {
      if ( (unsigned __int16)(v17 - 17) > 0xD3u )
      {
        LOWORD(v69) = v17;
        v70 = v18;
        goto LABEL_11;
      }
      LOWORD(v17) = word_4456580[v17 - 1];
      v39 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v67) )
      {
        v70 = v18;
        LOWORD(v69) = 0;
LABEL_5:
        v23 = (void *)sub_3007260((__int64)&v69);
        v24 = v25;
        v71[0] = v23;
        LODWORD(v25) = (_DWORD)v23;
        v71[1] = v24;
        goto LABEL_6;
      }
      LOWORD(v17) = sub_3009970((__int64)&v67, a4, v20, v21, v22);
    }
    LOWORD(v69) = v17;
    v70 = v39;
    if ( !(_WORD)v17 )
      goto LABEL_5;
LABEL_11:
    if ( (_WORD)v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      BUG();
    v25 = *(void **)&byte_444C4A0[16 * (unsigned __int16)v17 - 16];
LABEL_6:
    sub_C44740((__int64)&v69, v19, (unsigned int)v25);
    if ( a1 <= 0xC0 )
    {
      if ( a1 > 0xB3 )
      {
        switch ( a1 )
        {
          case 0xB4u:
            v40 = (_DWORD)v70 - 1;
            if ( (unsigned int)v70 <= 0x40 )
            {
              LOBYTE(v26) = (1LL << v40) - 1 == v69;
              return v26;
            }
            v26 = 0;
            if ( (*(_QWORD *)(v69 + 8LL * (v40 >> 6)) & (1LL << v40)) == 0 )
LABEL_56:
              LOBYTE(v26) = v40 == (unsigned int)sub_C445E0((__int64)&v69);
            goto LABEL_49;
          case 0xB5u:
            v41 = (_DWORD)v70 - 1;
            if ( (unsigned int)v70 <= 0x40 )
            {
              LOBYTE(v26) = 1LL << v41 == v69;
              return v26;
            }
            v26 = 0;
            if ( (*(_QWORD *)(v69 + 8LL * (v41 >> 6)) & (1LL << v41)) != 0 )
              LOBYTE(v26) = v41 == (unsigned int)sub_C44590((__int64)&v69);
            break;
          case 0xB6u:
          case 0xBAu:
            v40 = (unsigned int)v70;
            v26 = 1;
            if ( !(_DWORD)v70 )
              return v26;
            if ( (unsigned int)v70 > 0x40 )
              goto LABEL_56;
            LOBYTE(v26) = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v70) == v69;
            return v26;
          case 0xB7u:
          case 0xBBu:
          case 0xBCu:
            goto LABEL_47;
          case 0xBEu:
          case 0xBFu:
          case 0xC0u:
            goto LABEL_22;
          default:
            goto LABEL_57;
        }
        goto LABEL_49;
      }
      if ( a1 == 58 )
      {
        v42 = (int)v70;
        LOBYTE(v26) = v69 == 1;
        if ( (unsigned int)v70 <= 0x40 )
          return v26;
LABEL_68:
        LOBYTE(v26) = v42 - 1 == (unsigned int)sub_C444A0((__int64)&v69);
LABEL_49:
        if ( v69 )
        {
          v61 = v26;
          j_j___libc_free_0_0(v69);
          return v61;
        }
        return v26;
      }
      if ( a1 > 0x3A )
      {
        if ( a1 - 59 <= 1 )
        {
          if ( v8 != 1 )
            goto LABEL_18;
          v42 = (int)v70;
          if ( (unsigned int)v70 <= 0x40 )
          {
            LOBYTE(v26) = v69 == 1;
            return v26;
          }
          goto LABEL_68;
        }
        goto LABEL_57;
      }
      if ( a1 == 56 )
      {
LABEL_47:
        v28 = (int)v70;
        LOBYTE(v26) = v69 == 0;
        if ( (unsigned int)v70 <= 0x40 )
          return v26;
LABEL_48:
        LOBYTE(v26) = v28 == (unsigned int)sub_C444A0((__int64)&v69);
        goto LABEL_49;
      }
      if ( a1 == 57 )
      {
LABEL_22:
        if ( v8 != 1 )
        {
LABEL_18:
          v26 = 0;
          if ( (unsigned int)v70 <= 0x40 )
            return v26;
          goto LABEL_49;
        }
        v28 = (int)v70;
        if ( (unsigned int)v70 <= 0x40 )
        {
          LOBYTE(v26) = v69 == 0;
          return v26;
        }
        goto LABEL_48;
      }
    }
LABEL_57:
    if ( (unsigned int)v70 > 0x40 && v69 )
      j_j___libc_free_0_0(v69);
    return 0;
  }
  v29 = a4;
  v30 = sub_33E1790(a3, a4, 0, v12, v13, v14);
  v26 = 0;
  if ( v30 )
  {
    if ( a1 == 98 )
    {
      v51 = *(_QWORD *)(v30 + 96);
      v52 = sub_C33320();
      sub_C3B1B0((__int64)v71, 1.0);
      sub_C407B0(&v69, (__int64 *)v71, v52);
      sub_C338F0((__int64)v71);
      sub_C41640((__int64 *)&v69, *(_DWORD **)(v51 + 24), 1, (bool *)v71);
      v53 = *(void **)(v51 + 24);
      v54 = 0;
      if ( v53 == (void *)v69 )
      {
        v55 = v51 + 24;
        if ( v53 == sub_C33340() )
          v54 = sub_C3E590(v55, (__int64)&v69);
        else
          v54 = sub_C33D00(v55, (__int64)&v69);
      }
      v63 = v54;
      sub_91D830(&v69);
      return v63;
    }
    else if ( a1 <= 0x62 )
    {
      if ( a1 == 96 )
      {
        v49 = *(_QWORD *)(v30 + 96);
        if ( *(void **)(v49 + 24) == sub_C33340() )
          v50 = *(_QWORD *)(v49 + 32);
        else
          v50 = v49 + 24;
        v26 = 0;
        if ( (*(_BYTE *)(v50 + 20) & 7) == 3 )
        {
          v26 = 1;
          if ( (a2 & 0x80) == 0 )
            return (*(_BYTE *)(v50 + 20) & 8) != 0;
        }
      }
      else if ( a1 == 97 && v8 == 1 )
      {
        v37 = *(_QWORD *)(v30 + 96);
        v38 = *(void **)(v37 + 24) == sub_C33340() ? *(_QWORD *)(v37 + 32) : v37 + 24;
        v26 = 0;
        if ( (*(_BYTE *)(v38 + 20) & 7) == 3 )
        {
          v26 = 1;
          if ( (a2 & 0x80) == 0 )
            return ((*(_BYTE *)(v38 + 20) >> 3) ^ 1) & 1;
        }
      }
    }
    else if ( a1 == 99 )
    {
      if ( v8 == 1 )
      {
        v43 = *(_QWORD *)(v30 + 96);
        v44 = sub_C33320();
        sub_C3B1B0((__int64)v71, 1.0);
        sub_C407B0(&v69, (__int64 *)v71, v44);
        sub_C338F0((__int64)v71);
        sub_C41640((__int64 *)&v69, *(_DWORD **)(v43 + 24), 1, (bool *)v71);
        v45 = *(_QWORD *)(v43 + 24);
        v46 = v69;
        v47 = sub_C33340();
        v26 = 0;
        v48 = v47;
        if ( v45 == v46 )
        {
          v56 = v43 + 24;
          if ( (void *)v46 == v47 )
            LOBYTE(v57) = sub_C3E590(v56, (__int64)&v69);
          else
            LOBYTE(v57) = sub_C33D00(v56, (__int64)&v69);
          v46 = v69;
          v26 = v57;
        }
        if ( (void *)v46 == v48 )
        {
          if ( v70 )
          {
            for ( i = &v70[3 * *(v70 - 1)]; v70 != i; LOBYTE(v26) = v65 )
            {
              i -= 3;
              v65 = v26;
              sub_91D830(i);
            }
            v66 = v26;
            j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
            return v66;
          }
        }
        else
        {
          v62 = v26;
          sub_C338F0((__int64)&v69);
          return v62;
        }
      }
    }
    else
    {
      v59 = v30;
      if ( a1 - 279 <= 1 )
      {
        v31 = *(_QWORD *)(a3 + 48) + 16LL * (unsigned int)a4;
        v32 = *(_QWORD **)(v31 + 8);
        LOWORD(v69) = *(_WORD *)v31;
        v70 = v32;
        v33 = sub_300AC80((unsigned __int16 *)&v69, v29);
        v34 = sub_C33340();
        v35 = v34;
        if ( (a2 & 0x20) != 0 )
        {
          if ( (a2 & 0x40) != 0 )
          {
            if ( v33 == v34 )
              sub_C3C500(v71, (__int64)v34);
            else
              sub_C373C0(v71, (__int64)v33);
            if ( v71[0] == v35 )
              sub_C3CF90((__int64)v71, 0);
            else
              sub_C35910((__int64)v71, 0);
            v36 = v59;
          }
          else
          {
            if ( v33 == v34 )
              sub_C3C500(v71, (__int64)v34);
            else
              sub_C373C0(v71, (__int64)v33);
            if ( v71[0] == v35 )
              sub_C3CF20((__int64)v71, 0);
            else
              sub_C36EF0((_DWORD **)v71, 0);
            v36 = v59;
          }
        }
        else
        {
          if ( v33 == v34 )
            sub_C3C500(v71, (__int64)v34);
          else
            sub_C373C0(v71, (__int64)v33);
          if ( v71[0] == v35 )
            sub_C3D480((__int64)v71, 0, 0, 0);
          else
            sub_C36070((__int64)v71, 0, 0, 0);
          v36 = v59;
        }
        if ( a1 == 280 )
        {
          v64 = v36;
          if ( v35 == v71[0] )
            sub_C3CCB0((__int64)v71);
          else
            sub_C34440((unsigned __int8 *)v71);
          v36 = v64;
        }
        v60 = sub_33CA570(v36, v71);
        sub_91D830(v71);
        return v60;
      }
    }
  }
  return v26;
}
