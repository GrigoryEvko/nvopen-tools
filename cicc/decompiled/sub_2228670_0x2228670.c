// Function: sub_2228670
// Address: 0x2228670
//
_QWORD *__fastcall sub_2228670(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        int a5,
        _DWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        _DWORD *a10)
{
  unsigned int v10; // r15d
  __int64 v13; // rax
  _QWORD *v14; // r9
  __int64 v15; // r10
  unsigned __int64 *v16; // rdx
  void *v17; // rsp
  unsigned __int64 *v18; // r14
  unsigned __int64 v19; // rcx
  unsigned int v20; // r13d
  char v21; // bl
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // r12
  unsigned __int64 *v24; // r13
  __int64 v25; // r8
  _QWORD *v26; // r15
  char v27; // dl
  __int64 v28; // r9
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int *v31; // rax
  int v32; // eax
  bool v33; // zf
  _QWORD *v34; // rax
  _QWORD *v35; // r9
  int v36; // eax
  int *v37; // rax
  int v38; // eax
  unsigned int *v39; // rax
  unsigned int v40; // eax
  __int64 v42; // r13
  unsigned __int64 *v43; // r15
  __int64 v44; // rbx
  int v45; // r14d
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // r13
  __int64 v48; // rbx
  void *v49; // rsp
  int *v50; // rax
  int v51; // eax
  bool v52; // zf
  _QWORD *v53; // rax
  int *v54; // rax
  int v55; // eax
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  unsigned int *v59; // rax
  unsigned int v60; // eax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  _BYTE v65[4]; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v66; // [rsp+4h] [rbp-6Ch]
  _QWORD *v67; // [rsp+8h] [rbp-68h]
  _DWORD *v68; // [rsp+10h] [rbp-60h]
  unsigned __int64 *v69; // [rsp+18h] [rbp-58h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+28h] [rbp-48h]
  _QWORD *v72; // [rsp+30h] [rbp-40h]
  _QWORD *v73; // [rsp+38h] [rbp-38h]
  __int64 v74; // [rsp+80h] [rbp+10h]
  __int64 v75; // [rsp+80h] [rbp+10h]
  __int64 v76; // [rsp+80h] [rbp+10h]
  __int64 v77; // [rsp+80h] [rbp+10h]
  __int64 v78; // [rsp+80h] [rbp+10h]
  __int64 v79; // [rsp+80h] [rbp+10h]
  __int64 v80; // [rsp+80h] [rbp+10h]

  v10 = a3;
  v20 = a3;
  v70 = a3;
  v68 = a6;
  v72 = a4;
  v73 = a2;
  v13 = sub_2243120(a9 + 208);
  v14 = a2;
  v15 = a7;
  v16 = (unsigned __int64 *)v13;
  v17 = alloca(8 * a8 + 8);
  v18 = (unsigned __int64 *)v65;
  LOBYTE(v19) = v20 == -1;
  LOBYTE(v20) = v19 & (a2 != 0);
  if ( (_BYTE)v20 )
  {
    v54 = (int *)v73[2];
    if ( (unsigned __int64)v54 >= v73[3] )
    {
      v62 = *v73;
      v71 = (__int64)v16;
      v55 = (*(__int64 (__fastcall **)(_QWORD *))(v62 + 72))(v73);
      v15 = a7;
      v16 = (unsigned __int64 *)v71;
      v14 = v73;
    }
    else
    {
      v55 = *v54;
    }
    v19 = 0;
    if ( v55 == -1 )
    {
      v19 = v20;
      v14 = 0;
      LOBYTE(v20) = 0;
    }
  }
  LOBYTE(v73) = a5 == -1;
  v21 = (unsigned __int8)v73 & (a4 != 0);
  if ( v21 )
  {
    v50 = (int *)a4[2];
    if ( (unsigned __int64)v50 >= a4[3] )
    {
      v61 = *a4;
      v78 = v15;
      LOBYTE(v69) = v19;
      v71 = (__int64)v16;
      v72 = v14;
      v51 = (*(__int64 (__fastcall **)(_QWORD *))(v61 + 72))(a4);
      v15 = v78;
      v19 = (unsigned __int8)v69;
      v16 = (unsigned __int64 *)v71;
      v14 = v72;
    }
    else
    {
      v51 = *v50;
    }
    v52 = v51 == -1;
    v53 = 0;
    if ( !v52 )
      v53 = a4;
    v72 = v53;
    if ( !v52 )
      v21 = 0;
  }
  else
  {
    v21 = (char)v73;
  }
  if ( (_BYTE)v19 == v21 )
  {
    v22 = 0;
    v23 = 0;
    v24 = 0;
  }
  else
  {
    if ( (_BYTE)v20 )
    {
      v59 = (unsigned int *)v14[2];
      if ( (unsigned __int64)v59 >= v14[3] )
      {
        v64 = *v14;
        v80 = v15;
        v69 = v16;
        v71 = (__int64)v14;
        v60 = (*(__int64 (__fastcall **)(_QWORD *))(v64 + 72))(v14);
        v15 = v80;
        v16 = v69;
        v14 = (_QWORD *)v71;
        v19 = v60;
      }
      else
      {
        v19 = *v59;
        v60 = *v59;
      }
      if ( v60 == -1 )
        v14 = 0;
    }
    else
    {
      v19 = v10;
    }
    v22 = 2 * a8;
    if ( 2 * a8 )
    {
      v66 = v10;
      v42 = 0;
      v23 = 0;
      v43 = v16;
      v71 = 2 * a8;
      v44 = v15;
      v69 = (unsigned __int64 *)v65;
      v45 = v19;
      v67 = v14;
      do
      {
        while ( 1 )
        {
          a2 = (_QWORD *)**(unsigned int **)(v44 + 8 * v42);
          if ( (_DWORD)a2 == v45 || (*(unsigned int (__fastcall **)(unsigned __int64 *))(*v43 + 48))(v43) == v45 )
            break;
          if ( ++v42 == v71 )
            goto LABEL_63;
        }
        *((_DWORD *)v69 + v23++) = v42++;
      }
      while ( v42 != v71 );
LABEL_63:
      v15 = v44;
      v14 = v67;
      v10 = v66;
      v22 = 0;
      v18 = v69;
      v24 = 0;
      if ( v23 )
      {
        v46 = v67[2];
        if ( v46 >= v67[3] )
        {
          v63 = *v67;
          v79 = v15;
          v71 = (__int64)v67;
          (*(void (__fastcall **)(_QWORD *))(v63 + 80))(v67);
          v15 = v79;
          v14 = (_QWORD *)v71;
        }
        else
        {
          v67[2] = v46 + 4;
        }
        v71 = (__int64)v14;
        v47 = 0;
        v48 = v15;
        v49 = alloca(8 * v23 + 8);
        v69 = (unsigned __int64 *)v65;
        do
        {
          *(_QWORD *)&v65[8 * v47] = wcslen(*(const wchar_t **)(v48 + 8LL * *((int *)v18 + v47)));
          ++v47;
        }
        while ( v23 != v47 );
        v15 = v48;
        v14 = (_QWORD *)v71;
        v22 = v47;
        v10 = -1;
        v24 = v69;
        v23 = 1;
      }
    }
    else
    {
      v23 = 0;
      v24 = 0;
    }
  }
  v25 = v10;
  v26 = v14;
LABEL_7:
  LOBYTE(a2) = (_DWORD)v25 == -1;
  LOBYTE(v19) = (unsigned __int8)a2 & (v26 != 0);
  if ( (_BYTE)v19 )
  {
    v37 = (int *)v26[2];
    if ( (unsigned __int64)v37 >= v26[3] )
    {
      v58 = *v26;
      v77 = v15;
      LODWORD(v67) = v25;
      LOBYTE(v69) = (unsigned __int8)a2 & (v26 != 0);
      LOBYTE(v71) = (_DWORD)v25 == -1;
      v38 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64 *))(v58 + 72))(v26, a2, v16);
      v15 = v77;
      v25 = (unsigned int)v67;
      v19 = (unsigned __int8)v69;
      LOBYTE(a2) = v71;
    }
    else
    {
      v38 = *v37;
    }
    if ( v38 == -1 )
      v26 = 0;
    else
      v19 = 0;
  }
  else
  {
    v19 = (unsigned int)a2;
  }
  v27 = (unsigned __int8)v73 & (v72 != 0);
  if ( !v27 )
  {
    if ( (_BYTE)v19 == (_BYTE)v73 )
      goto LABEL_32;
    goto LABEL_11;
  }
  v31 = (int *)v72[2];
  if ( (unsigned __int64)v31 >= v72[3] )
  {
    v76 = v15;
    v66 = v25;
    v57 = *v72;
    LOBYTE(v67) = v19;
    LOBYTE(v69) = (unsigned __int8)v73 & (v72 != 0);
    LOBYTE(v71) = (_BYTE)a2;
    v32 = (*(__int64 (**)(void))(v57 + 72))();
    v15 = v76;
    v25 = v66;
    v19 = (unsigned __int8)v67;
    v27 = (char)v69;
    LOBYTE(a2) = v71;
  }
  else
  {
    v32 = *v31;
  }
  v33 = v32 == -1;
  v34 = 0;
  if ( !v33 )
    v34 = v72;
  v72 = v34;
  if ( !v33 )
    v27 = 0;
  if ( (_BYTE)v19 != v27 )
  {
LABEL_11:
    if ( !v26 || !(_BYTE)a2 )
    {
      v28 = (unsigned int)v25;
      if ( !v22 )
        goto LABEL_47;
      goto LABEL_14;
    }
    v39 = (unsigned int *)v26[2];
    if ( (unsigned __int64)v39 >= v26[3] )
    {
      v56 = *v26;
      v75 = v15;
      LODWORD(v71) = v25;
      v40 = (*(__int64 (__fastcall **)(_QWORD *))(v56 + 72))(v26);
      v15 = v75;
      v25 = (unsigned int)v71;
      v28 = v40;
    }
    else
    {
      v28 = *v39;
      v40 = *v39;
    }
    if ( v40 == -1 )
      v26 = 0;
    if ( v22 )
    {
LABEL_14:
      v29 = 0;
      a2 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v16 = &v24[v29];
          if ( *v16 > v23 )
            break;
          a2 = (_QWORD *)((char *)a2 + 1);
          ++v29;
LABEL_16:
          if ( v29 >= v22 )
            goto LABEL_20;
        }
        v19 = (unsigned __int64)v18 + 4 * v29;
        if ( *(_DWORD *)(*(_QWORD *)(v15 + 8LL * *(int *)v19) + 4 * v23) == (_DWORD)v28 )
        {
          ++v29;
          goto LABEL_16;
        }
        --v22;
        *(_DWORD *)v19 = *((_DWORD *)v18 + v22);
        v19 = v24[v22];
        *v16 = v19;
        if ( v29 >= v22 )
        {
LABEL_20:
          if ( (_QWORD *)v22 == a2 )
            goto LABEL_32;
          v30 = v26[2];
          if ( v30 >= v26[3] )
          {
            v74 = v15;
            (*(void (__fastcall **)(_QWORD *, _QWORD *, unsigned __int64 *, unsigned __int64, __int64, __int64))(*v26 + 80LL))(
              v26,
              a2,
              v16,
              v19,
              v25,
              v28);
            v15 = v74;
          }
          else
          {
            v26[2] = v30 + 4;
          }
          ++v23;
          v25 = 0xFFFFFFFFLL;
          goto LABEL_7;
        }
      }
    }
LABEL_47:
    v35 = v26;
LABEL_48:
    *a10 |= 4u;
    return v35;
  }
LABEL_32:
  v35 = v26;
  if ( v22 != 1 )
  {
    if ( v22 == 2 && (*v24 == v23 || v24[1] == v23) )
      goto LABEL_34;
    goto LABEL_48;
  }
  if ( *v24 != v23 )
    goto LABEL_48;
LABEL_34:
  v36 = *(_DWORD *)v18;
  if ( *(_DWORD *)v18 >= (int)a8 )
    v36 = *(_DWORD *)v18 - a8;
  *v68 = v36;
  return v35;
}
