// Function: sub_14D8AA0
// Address: 0x14d8aa0
//
__int64 __fastcall sub_14D8AA0(
        char *a1,
        char *a2,
        unsigned int a3,
        __int64 a4,
        __int64 *a5,
        unsigned __int64 a6,
        _BYTE *a7,
        __int64 a8,
        __int64 a9)
{
  unsigned __int64 v12; // r12
  _BYTE *v13; // rdi
  __int64 *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *v17; // rax
  unsigned int v18; // r15d
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rax
  char v35; // al
  __int64 v36; // rax
  _BYTE *v37; // rdi
  __int64 v38; // [rsp+8h] [rbp-1E8h]
  __int64 v41; // [rsp+30h] [rbp-1C0h]
  __int64 v42; // [rsp+40h] [rbp-1B0h]
  __int64 v43; // [rsp+40h] [rbp-1B0h]
  unsigned int v44; // [rsp+48h] [rbp-1A8h]
  __int64 v45; // [rsp+48h] [rbp-1A8h]
  _BYTE *v46; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-198h]
  _BYTE s[32]; // [rsp+60h] [rbp-190h] BYREF
  __int64 *v49; // [rsp+80h] [rbp-170h] BYREF
  __int64 v50; // [rsp+88h] [rbp-168h]
  _BYTE v51[32]; // [rsp+90h] [rbp-160h] BYREF
  _BYTE *v52; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v53; // [rsp+B8h] [rbp-138h]
  _BYTE v54[304]; // [rsp+C0h] [rbp-130h] BYREF

  v12 = *(_QWORD *)(a4 + 32);
  v13 = s;
  v46 = s;
  v47 = 0x400000000LL;
  if ( v12 > 4 )
  {
    sub_16CD150(&v46, s, v12, 8);
    v13 = v46;
  }
  LODWORD(v47) = v12;
  if ( 8LL * (unsigned int)v12 )
    memset(v13, 0, 8LL * (unsigned int)v12);
  v49 = (__int64 *)v51;
  v50 = 0x400000000LL;
  if ( a6 > 4 )
  {
    sub_16CD150(&v49, v51, a6, 8);
    v14 = v49;
  }
  else
  {
    v14 = (__int64 *)v51;
  }
  LODWORD(v50) = a6;
  if ( 8LL * (unsigned int)a6 )
    memset(v14, 0, 8LL * (unsigned int)a6);
  if ( a3 != 129 )
  {
    v15 = *(_QWORD *)(a4 + 32);
    if ( (_DWORD)v15 )
    {
      v42 = 0;
      v38 = (unsigned int)v15;
      v44 = a3 - 31;
      v41 = *(_QWORD *)(a4 + 24);
      while ( 1 )
      {
        v16 = 0;
        if ( (_DWORD)a6 )
        {
          v17 = a5;
          v18 = a3;
          v19 = v17;
          do
          {
            while ( v16 == 1 && ((v44 & 0xFFFFFFFD) == 0 || v18 == 147) )
            {
              v16 = 2;
              v49[1] = v19[1];
              if ( (unsigned int)a6 == 2 )
                goto LABEL_20;
            }
            v20 = sub_15A0A60(v19[v16], (unsigned int)v42);
            if ( !v20 )
              goto LABEL_24;
            v49[v16++] = v20;
          }
          while ( (unsigned int)a6 != v16 );
LABEL_20:
          v21 = v19;
          a3 = v18;
          a5 = v21;
        }
        v22 = sub_14D1BC0(a1, a2, a3, v41, v49, (unsigned int)v50, a8, a9);
        if ( !v22 )
          break;
        *(_QWORD *)&v46[8 * v42++] = v22;
        if ( v38 == v42 )
          goto LABEL_23;
      }
LABEL_24:
      v23 = 0;
    }
    else
    {
LABEL_23:
      v23 = sub_15A01B0(v46, (unsigned int)v47);
    }
    goto LABEL_25;
  }
  v25 = a5[2];
  v43 = a5[3];
  v26 = sub_14D8290(*a5, a4, a7);
  v27 = *(_QWORD *)(a4 + 32);
  v23 = v26;
  v52 = v54;
  v53 = 0x2000000000LL;
  if ( (_DWORD)v27 )
  {
    v28 = 0;
    while ( 1 )
    {
      v31 = sub_15A0A60(v25, v28);
      if ( !v31 )
      {
LABEL_63:
        v27 = (unsigned int)v53;
        if ( *(_QWORD *)(a4 + 32) != (unsigned int)v53 )
        {
LABEL_49:
          v23 = 0;
          goto LABEL_50;
        }
        v37 = v52;
        goto LABEL_65;
      }
      v32 = sub_15A0A60(v43, v28);
      v33 = v32;
      if ( v23 )
      {
        v45 = sub_15A0A60(v23, v28);
        if ( *(_BYTE *)(v31 + 16) != 9 )
        {
          if ( (unsigned __int8)sub_1593BB0(v31) )
            goto LABEL_48;
          goto LABEL_56;
        }
        if ( !v33 )
        {
          if ( !v45 )
            goto LABEL_49;
          v34 = (unsigned int)v53;
          if ( (unsigned int)v53 >= HIDWORD(v53) )
          {
            sub_16CD150(&v52, v54, 0, 8);
            v34 = (unsigned int)v53;
          }
          *(_QWORD *)&v52[8 * v34] = v45;
          LODWORD(v53) = v53 + 1;
          if ( (unsigned __int8)sub_1593BB0(v31) )
            goto LABEL_49;
          goto LABEL_56;
        }
      }
      else
      {
        if ( *(_BYTE *)(v31 + 16) != 9 )
        {
          if ( !(unsigned __int8)sub_1593BB0(v31) )
          {
            sub_15962C0(v31);
            goto LABEL_50;
          }
LABEL_48:
          if ( !v33 )
            goto LABEL_49;
LABEL_38:
          v30 = (unsigned int)v53;
          if ( (unsigned int)v53 >= HIDWORD(v53) )
          {
            sub_16CD150(&v52, v54, 0, 8);
            v30 = (unsigned int)v53;
          }
          *(_QWORD *)&v52[8 * v30] = v33;
          LODWORD(v53) = v53 + 1;
          goto LABEL_41;
        }
        if ( !v32 )
          goto LABEL_49;
        v45 = 0;
      }
      v29 = (unsigned int)v53;
      if ( (unsigned int)v53 >= HIDWORD(v53) )
      {
        sub_16CD150(&v52, v54, 0, 8);
        v29 = (unsigned int)v53;
      }
      *(_QWORD *)&v52[8 * v29] = v33;
      LODWORD(v53) = v53 + 1;
      if ( (unsigned __int8)sub_1593BB0(v31) )
        goto LABEL_38;
LABEL_56:
      v35 = sub_15962C0(v31);
      if ( !v45 || v35 != 1 )
        goto LABEL_49;
      v36 = (unsigned int)v53;
      if ( (unsigned int)v53 >= HIDWORD(v53) )
      {
        sub_16CD150(&v52, v54, 0, 8);
        v36 = (unsigned int)v53;
      }
      *(_QWORD *)&v52[8 * v36] = v45;
      LODWORD(v53) = v53 + 1;
LABEL_41:
      if ( (_DWORD)v27 == ++v28 )
        goto LABEL_63;
    }
  }
  v37 = v54;
  v23 = 0;
  if ( !v27 )
  {
LABEL_65:
    v23 = sub_15A01B0(v37, v27);
LABEL_50:
    if ( v52 != v54 )
      _libc_free((unsigned __int64)v52);
  }
LABEL_25:
  if ( v49 != (__int64 *)v51 )
    _libc_free((unsigned __int64)v49);
  if ( v46 != s )
    _libc_free((unsigned __int64)v46);
  return v23;
}
