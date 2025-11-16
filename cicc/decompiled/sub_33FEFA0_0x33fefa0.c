// Function: sub_33FEFA0
// Address: 0x33fefa0
//
__int64 __fastcall sub_33FEFA0(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 v10; // r15
  __int64 v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rcx
  __int64 *v21; // r11
  bool v22; // cl
  __int64 v23; // rdx
  void *v24; // r8
  __int64 v25; // rax
  bool v26; // cl
  void *v27; // r8
  __int64 *v28; // rax
  char v29; // al
  int v30; // eax
  void *v31; // rbx
  void *v32; // rsi
  _QWORD *v33; // rax
  __int64 *v34; // rsi
  _DWORD *v35; // rax
  __int64 *v36; // rdx
  bool v37; // dl
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r14
  __int64 v41; // r14
  void *v42; // [rsp+8h] [rbp-B8h]
  void *v43; // [rsp+10h] [rbp-B0h]
  void *v44; // [rsp+10h] [rbp-B0h]
  __int64 v45; // [rsp+18h] [rbp-A8h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  __int64 v47; // [rsp+28h] [rbp-98h]
  __int64 v48; // [rsp+30h] [rbp-90h]
  bool v49; // [rsp+30h] [rbp-90h]
  __int64 v50; // [rsp+30h] [rbp-90h]
  __int64 v51; // [rsp+30h] [rbp-90h]
  bool v52; // [rsp+30h] [rbp-90h]
  __int64 v53; // [rsp+38h] [rbp-88h]
  bool v54; // [rsp+38h] [rbp-88h]
  bool v55; // [rsp+38h] [rbp-88h]
  __int64 v56; // [rsp+38h] [rbp-88h]
  __int64 v57; // [rsp+40h] [rbp-80h] BYREF
  __int64 v58; // [rsp+48h] [rbp-78h]
  void *v59; // [rsp+50h] [rbp-70h] BYREF
  __int64 *v60; // [rsp+58h] [rbp-68h]
  __int64 v61[10]; // [rsp+70h] [rbp-50h] BYREF

  v57 = a4;
  v58 = a5;
  if ( a9 != 2 )
    return 0;
  v10 = *a8;
  v45 = a8[1];
  v47 = a8[2];
  v46 = *a8;
  v13 = a8[3];
  v14 = sub_33E1790(*a8, v45, 0, a4, v47, a6);
  v17 = sub_33E1790(v47, v13, 0, v15, v47, v16);
  LOBYTE(v20) = v14 != 0;
  v53 = v17;
  if ( !v17 || !v14 )
  {
LABEL_19:
    if ( a2 == 230 && (_BYTE)v20 )
    {
      v33 = sub_C33340();
      v34 = (__int64 *)(*(_QWORD *)(v14 + 96) + 24LL);
      if ( (_QWORD *)*v34 == v33 )
        sub_C3C790(v61, (_QWORD **)v34);
      else
        sub_C33EB0(v61, v34);
      v35 = sub_300AC80((unsigned __int16 *)&v57, (__int64)v34);
      sub_C41640(v61, v35, 1, (bool *)&v59);
      goto LABEL_37;
    }
    if ( a2 != 97 )
    {
      if ( a2 <= 0x61 )
      {
        if ( a2 != 96 )
          return 0;
      }
      else if ( a2 - 98 > 2 )
      {
        return 0;
      }
      goto LABEL_24;
    }
    v13 = v45;
    v39 = sub_33E1790(v10, v45, 1u, v20, v18, v19);
    if ( v39 )
    {
      v40 = *(_QWORD *)(v39 + 96);
      v31 = sub_C33340();
      if ( *(void **)(v40 + 24) == v31 )
      {
        v41 = *(_QWORD *)(v40 + 32);
        if ( (*(_BYTE *)(v41 + 20) & 7) == 3 )
        {
LABEL_69:
          v30 = *(_DWORD *)(v47 + 24);
          if ( (*(_BYTE *)(v41 + 20) & 8) != 0 )
          {
            if ( v30 != 51 )
            {
              if ( *(_DWORD *)(v46 + 24) == 51 )
                goto LABEL_28;
              return 0;
            }
            return sub_3288990(a1, (unsigned int)v57, v58);
          }
LABEL_25:
          if ( *(_DWORD *)(v46 + 24) != 51 )
          {
            if ( v30 == 51 )
            {
LABEL_27:
              v31 = sub_C33340();
LABEL_28:
              v32 = sub_300AC80((unsigned __int16 *)&v57, v13);
              if ( v32 == v31 )
                sub_C3C500(v61, (__int64)v31);
              else
                sub_C373C0(v61, (__int64)v32);
              if ( v31 == (void *)v61[0] )
                sub_C3D480((__int64)v61, 0, 0, 0);
              else
                sub_C36070((__int64)v61, 0, 0, 0);
LABEL_37:
              v50 = sub_33FE6E0(a1, v61, a3, v57, v58, 0, a7);
              sub_91D830(v61);
              return v50;
            }
            return 0;
          }
          if ( v30 != 51 )
            goto LABEL_27;
          return sub_3288990(a1, (unsigned int)v57, v58);
        }
      }
      else if ( (*(_BYTE *)(v40 + 44) & 7) == 3 )
      {
        v41 = v40 + 24;
        goto LABEL_69;
      }
    }
LABEL_24:
    v30 = *(_DWORD *)(v47 + 24);
    goto LABEL_25;
  }
  v48 = *(_QWORD *)(v14 + 96);
  v42 = sub_C33340();
  v21 = (__int64 *)(v48 + 24);
  if ( *(void **)(v48 + 24) == v42 )
  {
    sub_C3C790(&v59, (_QWORD **)v21);
    v24 = v42;
    v23 = v53;
    v22 = v14 != 0;
  }
  else
  {
    sub_C33EB0(&v59, v21);
    v22 = v14 != 0;
    v23 = v53;
    v24 = v42;
  }
  v25 = *(_QWORD *)(v23 + 96);
  v13 = v25 + 24;
  if ( a2 > 0x11E )
    goto LABEL_18;
  if ( a2 > 0x116 )
  {
    switch ( a2 )
    {
      case 0x117u:
        sub_969700(v61, &v59, (_QWORD *)v13);
        break;
      case 0x118u:
        sub_969910(v61, &v59, (_QWORD *)v13);
        break;
      case 0x11Bu:
        sub_969B00(v61, (__int64 *)&v59, (__int64 *)v13);
        break;
      case 0x11Cu:
        sub_969CF0(v61, (__int64 *)&v59, (__int64 *)v13);
        break;
      case 0x11Du:
        sub_33C9F70(v61, (__int64 *)&v59, v13);
        break;
      case 0x11Eu:
        sub_33CA140(v61, &v59, v13);
        break;
      default:
        goto LABEL_18;
    }
    v51 = sub_33FE6E0(a1, v61, a3, v57, v58, 0, a7);
    sub_91D830(v61);
    v38 = v51;
    goto LABEL_56;
  }
  if ( a2 <= 0x64 )
  {
    if ( a2 <= 0x5F )
      goto LABEL_18;
    v43 = v24;
    v49 = v22;
    switch ( a2 )
    {
      case 'a':
        if ( v24 == v59 )
        {
          sub_C3D820((__int64 *)&v59, v13, 1u);
          v27 = v43;
          v26 = v49;
        }
        else
        {
          sub_C3B1F0((__int64)&v59, v13, 1);
          v26 = v49;
          v27 = v43;
        }
        break;
      case 'b':
        if ( v24 == v59 )
        {
          sub_C3F5C0((__int64)&v59, (__int64 *)v13, 1u);
          v27 = v43;
          v26 = v49;
        }
        else
        {
          sub_C3B950((__int64)&v59, v13, 1);
          v26 = v49;
          v27 = v43;
        }
        break;
      case 'c':
        if ( v24 == v59 )
        {
          sub_C3EF50(&v59, v13, 1u);
          v27 = v43;
          v26 = v49;
        }
        else
        {
          sub_C3B6C0((__int64)&v59, v13, 1);
          v26 = v49;
          v27 = v43;
        }
        break;
      case 'd':
        if ( v24 == v59 )
        {
          sub_C3EC80(&v59, v13);
          v27 = v43;
          v26 = v49;
        }
        else
        {
          sub_C3BE30(&v59, (__int64 *)v13);
          v26 = v49;
          v27 = v43;
        }
        break;
      default:
        v52 = v22;
        v44 = v24;
        if ( v24 == v59 )
        {
          sub_C3D800((__int64 *)&v59, v13, 1u);
          v27 = v44;
          v26 = v52;
        }
        else
        {
          sub_C3ADF0((__int64)&v59, v13, 1);
          v26 = v52;
          v27 = v44;
        }
        break;
    }
    v28 = (__int64 *)&v59;
    v54 = v26;
    if ( v27 == v59 )
      v28 = v60;
    if ( (*((_BYTE *)v28 + 20) & 7) == 1 )
    {
      v29 = sub_C33750((__int64)v59);
      v22 = v54;
      if ( v29 )
        goto LABEL_18;
    }
    goto LABEL_55;
  }
  if ( a2 != 152 )
  {
LABEL_18:
    v55 = v22;
    sub_91D830(&v59);
    v20 = v55;
    goto LABEL_19;
  }
  v36 = (__int64 *)&v59;
  if ( v24 == v59 )
    v36 = v60;
  v37 = (*((_BYTE *)v36 + 20) & 8) != 0;
  if ( v24 == *(void **)(v25 + 24) )
    v13 = *(_QWORD *)(v25 + 32);
  if ( ((*(_BYTE *)(v13 + 20) & 8) != 0) != v37 )
  {
    if ( v24 == v59 )
      sub_C3CCB0((__int64)&v59);
    else
      sub_C34440((unsigned __int8 *)&v59);
  }
LABEL_55:
  v38 = sub_33FE6E0(a1, (__int64 *)&v59, a3, v57, v58, 0, a7);
LABEL_56:
  v56 = v38;
  sub_91D830(&v59);
  return v56;
}
