// Function: sub_11875C0
// Address: 0x11875c0
//
__int64 __fastcall sub_11875C0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  __int64 v5; // r14
  _QWORD *v7; // rax
  __int16 v8; // r12
  __int64 v9; // r15
  _BYTE *v10; // rbx
  unsigned int v12; // r12d
  unsigned int v13; // edx
  bool v14; // zf
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  char v17; // al
  unsigned int v18; // eax
  unsigned int v19; // edx
  unsigned __int64 v20; // rcx
  _QWORD *v21; // rcx
  __int64 *v22; // rdi
  __int64 v23; // rbx
  __int64 v24; // rax
  _QWORD *v25; // rax
  char v26; // al
  unsigned __int8 *v27; // rdx
  __int64 v28; // rdx
  __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  bool v31; // al
  unsigned int v32; // edx
  unsigned __int64 v33; // rcx
  const void **v34; // rax
  unsigned int v35; // [rsp+8h] [rbp-C8h]
  int v36; // [rsp+10h] [rbp-C0h]
  _QWORD *v37; // [rsp+18h] [rbp-B8h]
  _QWORD *v38; // [rsp+20h] [rbp-B0h]
  __int64 v39; // [rsp+30h] [rbp-A0h] BYREF
  const void **v40; // [rsp+38h] [rbp-98h] BYREF
  __int64 v41; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-88h]
  __int64 v43; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v44; // [rsp+58h] [rbp-78h]
  _QWORD *v45; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v46; // [rsp+68h] [rbp-68h]
  _QWORD *v47; // [rsp+70h] [rbp-60h] BYREF
  __int64 *v48; // [rsp+78h] [rbp-58h] BYREF
  _QWORD *v49; // [rsp+80h] [rbp-50h]
  __int16 v50; // [rsp+90h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 16);
  if ( !v4 )
    return 0;
  v5 = *(_QWORD *)(v4 + 8);
  if ( v5 )
    return 0;
  v7 = *(_QWORD **)(a1 - 64);
  v8 = *(_WORD *)(a1 + 2);
  v9 = (__int64)a2;
  v47 = 0;
  v10 = (_BYTE *)a3;
  v38 = v7;
  v12 = v8 & 0x3F;
  v37 = *(_QWORD **)(a1 - 32);
  if ( (unsigned __int8)sub_995B10(&v47, a3) )
  {
    v12 = sub_B52870(v12);
    v9 = (__int64)v10;
    v10 = a2;
  }
  v47 = 0;
  if ( !(unsigned __int8)sub_995B10(&v47, v9) )
    return 0;
  if ( v12 == 32 )
  {
    v14 = *v10 == 42;
    v48 = 0;
    v47 = v38;
    if ( !v14 )
      return v5;
    if ( v38 != *((_QWORD **)v10 - 8) )
      return v5;
    if ( !(unsigned __int8)sub_993A50(&v48, *((_QWORD *)v10 - 4)) )
      return v5;
    v45 = 0;
    if ( !(unsigned __int8)sub_995B10(&v45, (__int64)v37) )
      return v5;
    v23 = (__int64)v38;
    HIDWORD(v45) = 0;
    v50 = 257;
    v24 = sub_AD64C0(v38[1], 1, 0);
    return sub_B33C40(a4, 0x167u, v23, v24, (__int64)v45, (__int64)&v47);
  }
  v13 = v12 - 34;
  if ( v12 - 34 <= 1 )
  {
    v14 = *v10 == 42;
    LOBYTE(v49) = 1;
    v47 = v38;
    v48 = &v39;
    if ( v14 && v38 == *((_QWORD **)v10 - 8) )
    {
      v17 = sub_991580((__int64)&v48, *((_QWORD *)v10 - 4));
      v13 = v12 - 34;
      if ( v17 )
      {
        sub_9865C0((__int64)&v43, v39);
        v18 = v44;
        v19 = v12 - 34;
        if ( v44 > 0x40 )
        {
          sub_C43D10((__int64)&v43);
          v18 = v44;
          v21 = (_QWORD *)v43;
          v19 = v12 - 34;
        }
        else
        {
          v20 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
          if ( !v44 )
            v20 = 0;
          v21 = (_QWORD *)(~v43 & v20);
          v43 = (__int64)v21;
        }
        LODWORD(v46) = v18;
        v41 = (__int64)&v45;
        v35 = v19;
        v45 = v21;
        v44 = 0;
        if ( sub_117F800((const void ***)&v41, (__int64)v37) )
        {
LABEL_34:
          sub_969240((__int64 *)&v45);
          v22 = &v43;
LABEL_35:
          sub_969240(v22);
          v23 = (__int64)v38;
          HIDWORD(v45) = 0;
          v50 = 257;
          v24 = sub_AD8D80(v38[1], v39);
          return sub_B33C40(a4, 0x167u, v23, v24, (__int64)v45, (__int64)&v47);
        }
        sub_969240((__int64 *)&v45);
        sub_969240(&v43);
        v13 = v35;
      }
    }
  }
  if ( v12 == 34 )
  {
    v14 = *v10 == 42;
    LOBYTE(v49) = 1;
    v47 = v38;
    v48 = &v39;
    if ( !v14 || v38 != *((_QWORD **)v10 - 8) || !(unsigned __int8)sub_991580((__int64)&v48, *((_QWORD *)v10 - 4)) )
      goto LABEL_43;
    sub_9865C0((__int64)&v41, v39);
    v32 = v42;
    if ( v42 > 0x40 )
    {
      sub_C43D10((__int64)&v41);
      v32 = v42;
      v34 = (const void **)v41;
    }
    else
    {
      v33 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v42;
      if ( !v42 )
        v33 = 0;
      v34 = (const void **)(v33 & ~v41);
      v41 = (__int64)v34;
    }
    v44 = v32;
    v43 = (__int64)v34;
    v42 = 0;
    sub_C46F20((__int64)&v43, 1u);
    v40 = (const void **)&v45;
    LODWORD(v46) = v44;
    v45 = (_QWORD *)v43;
    v44 = 0;
    if ( !sub_117F800(&v40, (__int64)v37) || sub_986760(v39) )
    {
      sub_969240((__int64 *)&v45);
      sub_969240(&v43);
      sub_969240(&v41);
      goto LABEL_43;
    }
    sub_969240((__int64 *)&v45);
    sub_969240(&v43);
    v22 = &v41;
    goto LABEL_35;
  }
  if ( v12 != 35 )
  {
    if ( v13 > 1 )
      goto LABEL_13;
    goto LABEL_43;
  }
  v14 = *v10 == 42;
  LOBYTE(v49) = 1;
  v47 = v38;
  v48 = &v39;
  if ( v14 && v38 == *((_QWORD **)v10 - 8) && (unsigned __int8)sub_991580((__int64)&v48, *((_QWORD *)v10 - 4)) )
  {
    sub_9865C0((__int64)&v43, v39);
    if ( v44 > 0x40 )
    {
      sub_C43D10((__int64)&v43);
    }
    else
    {
      v30 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v44;
      if ( !v44 )
        v30 = 0;
      v43 = v30 & ~v43;
    }
    sub_C46250((__int64)&v43);
    v41 = (__int64)&v45;
    LODWORD(v46) = v44;
    v45 = (_QWORD *)v43;
    v44 = 0;
    if ( !sub_117F800((const void ***)&v41, (__int64)v37)
      || (*(_DWORD *)(v39 + 8) <= 0x40u
        ? (v31 = *(_QWORD *)v39 == 0)
        : (v36 = *(_DWORD *)(v39 + 8), v31 = v36 == (unsigned int)sub_C444A0(v39)),
          v31) )
    {
      sub_969240((__int64 *)&v45);
      sub_969240(&v43);
      goto LABEL_43;
    }
    goto LABEL_34;
  }
LABEL_43:
  v12 = sub_B52F50(v12);
  v25 = v38;
  v38 = v37;
  v37 = v25;
LABEL_13:
  if ( v12 - 36 > 1 )
    return 0;
  v45 = 0;
  v46 = &v41;
  if ( (unsigned __int8)sub_996420(&v45, 30, (unsigned __int8 *)v38) )
  {
    v14 = *v10 == 42;
    v47 = (_QWORD *)v41;
    v48 = &v43;
    if ( !v14 )
    {
      v41 = (__int64)v38;
      v43 = (__int64)v37;
      goto LABEL_17;
    }
    if ( (unsigned __int8)sub_11783C0((__int64)&v47, (__int64)v10) && (_QWORD *)v43 == v37 )
    {
      HIDWORD(v45) = 0;
      v50 = 257;
      return sub_B33C40(a4, 0x167u, v41, (__int64)v37, (unsigned int)v45, (__int64)&v47);
    }
  }
  v47 = 0;
  v26 = *v10;
  v41 = (__int64)v38;
  v43 = (__int64)v37;
  v48 = v38;
  v49 = v37;
  if ( v26 == 42 )
  {
    if ( sub_1187280((__int64)&v47, 30, *((unsigned __int8 **)v10 - 8)) )
    {
      v29 = *((_QWORD *)v10 - 4);
      v27 = (unsigned __int8 *)v29;
      if ( (_QWORD *)v29 == v49 )
      {
        v28 = *((_QWORD *)v10 - 8);
        goto LABEL_50;
      }
    }
    else
    {
      v27 = (unsigned __int8 *)*((_QWORD *)v10 - 4);
    }
    if ( !sub_1187280((__int64)&v47, 30, v27) )
      goto LABEL_17;
    v28 = *((_QWORD *)v10 - 8);
    if ( (_QWORD *)v28 != v49 )
      goto LABEL_17;
    v29 = *((_QWORD *)v10 - 4);
LABEL_50:
    HIDWORD(v45) = 0;
    v50 = 257;
    return sub_B33C40(a4, 0x167u, v28, v29, (unsigned int)v45, (__int64)&v47);
  }
LABEL_17:
  if ( v12 == 36 )
  {
    v47 = v37;
    v48 = &v43;
    if ( *(_BYTE *)v38 == 42 && (unsigned __int8)sub_11783C0((__int64)&v47, (__int64)v38) && *v10 == 42 )
    {
      if ( (v15 = (_QWORD *)*((_QWORD *)v10 - 8), v16 = (_QWORD *)*((_QWORD *)v10 - 4), v37 == v15)
        && (_QWORD *)v43 == v16
        || v16 == v37 && (_QWORD *)v43 == v15 )
      {
        HIDWORD(v45) = 0;
        v50 = 257;
        return sub_B33C40(a4, 0x167u, (__int64)v37, v43, (unsigned int)v45, (__int64)&v47);
      }
    }
  }
  return v5;
}
