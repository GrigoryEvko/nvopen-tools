// Function: sub_2237F70
// Address: 0x2237f70
//
_QWORD *__fastcall sub_2237F70(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        _QWORD *a4,
        __int64 a5,
        __int64 a6,
        int *a7,
        _BYTE *a8)
{
  int v8; // edx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  _QWORD *v13; // r13
  __int64 v14; // rax
  bool v15; // bp
  __int64 v16; // rsi
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // r15
  bool v20; // r14
  unsigned int v21; // ebx
  char v22; // r12
  __int64 v23; // r8
  unsigned __int64 v24; // rax
  bool v25; // di
  char v26; // r12
  __int64 v27; // r8
  __int64 v28; // r9
  _BYTE *v29; // rax
  int v30; // eax
  int v31; // eax
  unsigned __int8 v32; // [rsp+Dh] [rbp-7Bh]
  unsigned __int8 v33; // [rsp+Fh] [rbp-79h]
  _QWORD *v34; // [rsp+20h] [rbp-68h] BYREF
  __int64 v35; // [rsp+28h] [rbp-60h]
  _QWORD *v36; // [rsp+30h] [rbp-58h] BYREF
  __int64 v37; // [rsp+38h] [rbp-50h]
  unsigned __int64 v38[8]; // [rsp+48h] [rbp-40h] BYREF

  v36 = a2;
  v37 = a3;
  v34 = a4;
  v35 = a5;
  if ( (*(_BYTE *)(a6 + 24) & 1) == 0 )
  {
    v38[0] = -1;
    v36 = sub_2237410(a1, a2, a3, a4, a5, a6, a7, v38);
    LODWORD(v37) = v8;
    if ( v38[0] > 1 )
    {
      *a8 = 1;
      *a7 = 4;
      if ( sub_2233E50((__int64)&v36, (__int64)&v34) )
        *a7 |= 2u;
    }
    else
    {
      *a8 = v38[0];
      *a8 &= 1u;
    }
    return v36;
  }
  v16 = a6 + 208;
  v10 = sub_2232A70((__int64)v38, (__int64 *)(a6 + 208));
  v12 = *(_QWORD *)(v10 + 64);
  v13 = (_QWORD *)v10;
  v14 = *(_QWORD *)(v10 + 48);
  v15 = v12 == 0;
  LOBYTE(v16) = v14 == 0;
  v17 = v14 | v12;
  if ( !v17 )
    goto LABEL_42;
  LODWORD(v18) = v37;
  v19 = 0;
  v20 = 1;
  v21 = 1;
  while ( 1 )
  {
    v26 = (_DWORD)v18 == -1;
    LOBYTE(v18) = v26 & (v36 != 0);
    v27 = (unsigned int)v18;
    if ( (_BYTE)v18 )
    {
      v18 = v36[3];
      v26 = 0;
      if ( v36[2] >= v18 )
      {
        v32 = v27;
        LODWORD(v18) = (*(__int64 (**)(void))(*v36 + 72LL))();
        v27 = v32;
        v16 = (unsigned __int8)v16;
        if ( (_DWORD)v18 == -1 )
        {
          v36 = 0;
          v26 = v32;
        }
      }
    }
    LOBYTE(v27) = (_DWORD)v35 == -1;
    LOBYTE(v18) = v27 & (v34 != 0);
    v28 = (unsigned int)v18;
    if ( (_BYTE)v18 )
    {
      v27 = 0;
      if ( v34[2] >= v34[3] )
      {
        v33 = v18;
        v30 = (*(__int64 (**)(void))(*v34 + 72LL))();
        v27 = 0;
        v16 = (unsigned __int8)v16;
        v28 = v33;
        if ( v30 == -1 )
        {
          v34 = 0;
          v27 = v33;
        }
      }
    }
    if ( (_BYTE)v27 == v26 )
      break;
    LOBYTE(v29) = v37;
    if ( (_DWORD)v37 == -1 && v36 )
    {
      v29 = (_BYTE *)v36[2];
      if ( (unsigned __int64)v29 >= v36[3] )
      {
        LODWORD(v29) = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v36 + 72LL))(
                         v36,
                         v16,
                         v11,
                         v17,
                         v27,
                         v28);
        v16 = (unsigned __int8)v16;
        if ( (_DWORD)v29 == -1 )
          v36 = 0;
      }
      else
      {
        LOBYTE(v29) = *v29;
      }
    }
    if ( !v15 )
    {
      v17 = v13[7];
      LOBYTE(v21) = *(_BYTE *)(v17 + v19) == (unsigned __int8)v29;
    }
    if ( !(_BYTE)v21 )
    {
      if ( (_BYTE)v16 )
      {
        if ( v20 )
        {
          if ( v13[6] == v19 && v19 )
          {
            *a8 = 1;
            v31 = 0;
            goto LABEL_62;
          }
          goto LABEL_42;
        }
        goto LABEL_53;
      }
LABEL_8:
      v17 = v13[5];
      v20 = *(_BYTE *)(v17 + v19) == (unsigned __int8)v29;
      goto LABEL_9;
    }
    if ( !(_BYTE)v16 )
      goto LABEL_8;
LABEL_9:
    v22 = v15 && !v20;
    if ( v22 )
    {
      if ( (_BYTE)v21 )
      {
        if ( v13[8] == v19 && v19 )
        {
          *a8 = 0;
          v31 = 0;
          goto LABEL_62;
        }
LABEL_42:
        *a8 = 0;
LABEL_43:
        *a7 = 4;
        return v36;
      }
LABEL_53:
      *a8 = 0;
      *a7 = 4;
      return v36;
    }
    v23 = v21;
    LOBYTE(v23) = v20 | v21;
    if ( !(v20 | (unsigned __int8)v21) )
      goto LABEL_53;
    ++v19;
    v24 = v36[2];
    if ( v24 >= v36[3] )
    {
      (*(void (__fastcall **)(_QWORD *, __int64, __int64, __int64, __int64, __int64))(*v36 + 80LL))(
        v36,
        v16,
        v11,
        v17,
        v23,
        v28);
      LOBYTE(v23) = v20 | v21;
    }
    else
    {
      v36[2] = v24 + 1;
    }
    LODWORD(v37) = -1;
    if ( !(_BYTE)v21 )
    {
      v15 = v23;
LABEL_30:
      v20 = v23;
      LOBYTE(v16) = v13[6] <= v19;
      v25 = v16 & v15;
      goto LABEL_16;
    }
    v15 = v13[8] <= v19;
    if ( v20 )
      goto LABEL_30;
    v25 = v13[8] <= v19;
    v16 = v21;
LABEL_16:
    LODWORD(v18) = -1;
    if ( v25 )
      goto LABEL_45;
  }
  v22 = 1;
LABEL_45:
  if ( (_BYTE)v21 && v13[8] == v19 && v19 )
  {
    *a8 = 0;
    if ( v20 && v13[6] == v19 )
      goto LABEL_43;
  }
  else
  {
    if ( !v20 || v13[6] != v19 || !v19 )
    {
      *a8 = 0;
      if ( !v22 )
        goto LABEL_43;
      *a7 = 6;
      return v36;
    }
    *a8 = 1;
  }
  v31 = 2 * (v22 != 0);
LABEL_62:
  *a7 = v31;
  return v36;
}
