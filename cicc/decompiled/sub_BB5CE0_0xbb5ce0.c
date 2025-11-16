// Function: sub_BB5CE0
// Address: 0xbb5ce0
//
__int64 __fastcall sub_BB5CE0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 (__fastcall *a7)(__int64, __int64, __int64 *),
        __int64 a8)
{
  char v11; // al
  _QWORD *v12; // rdx
  signed __int64 v13; // rdi
  unsigned __int64 v14; // r9
  unsigned __int64 v15; // r8
  __int64 v16; // rdi
  char v17; // al
  _QWORD *v18; // rdx
  __int64 v19; // r13
  unsigned __int64 v20; // rsi
  char v21; // cl
  __int64 v22; // r8
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rax
  __int64 v32; // r13
  unsigned __int8 v33; // cl
  unsigned __int64 v34; // rax
  unsigned int v35; // r10d
  __int64 v36; // r9
  int v37; // eax
  bool v38; // al
  _QWORD *v39; // r13
  __int64 v40; // rax
  unsigned __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rdx
  char v44; // r13
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // r13
  _BYTE *v49; // rax
  __int64 v50; // [rsp+0h] [rbp-D0h]
  _QWORD *v51; // [rsp+10h] [rbp-C0h]
  unsigned int v52; // [rsp+18h] [rbp-B8h]
  unsigned __int8 v53; // [rsp+1Fh] [rbp-B1h]
  __int64 v55; // [rsp+28h] [rbp-A8h]
  char v56; // [rsp+28h] [rbp-A8h]
  __int64 v57; // [rsp+28h] [rbp-A8h]
  char v58; // [rsp+30h] [rbp-A0h]
  __int64 v59; // [rsp+30h] [rbp-A0h]
  __int64 v60; // [rsp+30h] [rbp-A0h]
  _QWORD *v61; // [rsp+38h] [rbp-98h]
  char v62; // [rsp+4Fh] [rbp-81h] BYREF
  unsigned __int8 *v63[2]; // [rsp+50h] [rbp-80h] BYREF
  _QWORD *v64; // [rsp+60h] [rbp-70h] BYREF
  __int64 v65; // [rsp+68h] [rbp-68h]
  __int64 v66; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v67; // [rsp+78h] [rbp-58h]
  unsigned __int64 v68; // [rsp+80h] [rbp-50h] BYREF
  __int64 v69; // [rsp+88h] [rbp-48h]
  __int64 v70; // [rsp+90h] [rbp-40h] BYREF
  __int64 v71; // [rsp+98h] [rbp-38h]

  v11 = sub_BCAC40(a1, 8);
  v12 = a2;
  v53 = v11 & (a7 == 0 && a3 != 0);
  if ( !v53 )
  {
    v62 = 0;
    v64 = a2;
    v63[0] = (unsigned __int8 *)a5;
    v13 = a1 & 0xFFFFFFFFFFFFFFF9LL | 4;
    v63[1] = (unsigned __int8 *)&v62;
    v61 = &a2[a3];
    v65 = a1 & 0xFFFFFFFFFFFFFFF9LL | 4;
    if ( v61 == a2 )
      return 1;
    while ( 1 )
    {
      v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = v13 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v13 )
        goto LABEL_39;
      v16 = (v13 >> 1) & 3;
      if ( v16 != 2 )
        break;
      if ( !v14 )
        goto LABEL_39;
LABEL_10:
      v17 = sub_BCEA30(v15);
      v18 = v64;
      v19 = v65;
      v20 = 0;
      v21 = v17;
      v22 = *v64;
      if ( v65 )
      {
        v20 = v65 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v65 & 6) != 0 )
          v20 = 0;
      }
      if ( *(_BYTE *)v22 != 17 || *(_BYTE *)(*(_QWORD *)(v22 + 8) + 8LL) != 12 )
      {
        if ( !a7 || v20 || v17 )
          return v53;
        v66 = 0;
        v67 = 1;
        if ( !a7(a8, v22, &v66) )
        {
LABEL_73:
          if ( v67 > 0x40 )
          {
            v46 = v66;
            if ( v66 )
              goto LABEL_75;
          }
          return v53;
        }
        v62 = 1;
        v23 = v65 & 0xFFFFFFFFFFFFFFF8LL;
        v24 = v65 & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v65 )
          goto LABEL_41;
        v25 = (v65 >> 1) & 3;
        if ( v25 != 2 )
        {
          if ( v25 != 1 || !v23 )
            goto LABEL_41;
          v24 = *(_QWORD *)(v23 + 24);
          goto LABEL_42;
        }
        if ( v23 )
          goto LABEL_22;
LABEL_41:
        v24 = sub_BCBAE0(v23, *v64);
        if ( ((v65 >> 1) & 3) == 1 )
        {
LABEL_42:
          v70 = sub_9208B0(a4, v24);
          v71 = v26;
          v27 = (unsigned __int64)(v70 + 7) >> 3;
        }
        else
        {
LABEL_22:
          v58 = sub_AE5020(a4, v24);
          v70 = sub_9208B0(a4, v24);
          v71 = v26;
          v27 = ((1LL << v58) + ((unsigned __int64)(v70 + 7) >> 3) - 1) >> v58 << v58;
        }
        LOBYTE(v69) = v26;
        v68 = v27;
        v28 = sub_CA1930(&v68);
        LODWORD(v71) = v67;
        if ( v67 > 0x40 )
        {
          v55 = v28;
          sub_C43780(&v70, &v66);
          v28 = v55;
        }
        else
        {
          v70 = v66;
        }
        v29 = sub_BB50D0(v63, (__int64)&v70, v28);
        if ( (unsigned int)v71 > 0x40 && v70 )
          j_j___libc_free_0_0(v70);
        if ( !v29 )
          goto LABEL_73;
        if ( v67 > 0x40 && v66 )
          j_j___libc_free_0_0(v66);
LABEL_32:
        v19 = v65;
        v18 = v64;
        goto LABEL_33;
      }
      v35 = *(_DWORD *)(v22 + 32);
      v36 = v22 + 24;
      if ( v35 <= 0x40 )
      {
        v38 = *(_QWORD *)(v22 + 24) == 0;
      }
      else
      {
        v52 = *(_DWORD *)(v22 + 32);
        v50 = *v64;
        v51 = v64;
        v56 = v17;
        v59 = v22 + 24;
        v37 = sub_C444A0(v22 + 24);
        v35 = v52;
        v36 = v59;
        v21 = v56;
        v18 = v51;
        v22 = v50;
        v38 = v52 == v37;
      }
      if ( !v38 )
      {
        if ( v21 )
          return v53;
        if ( v20 )
        {
          v39 = *(_QWORD **)(v22 + 24);
          if ( v35 > 0x40 )
            v39 = (_QWORD *)*v39;
          v40 = 16LL * (unsigned int)v39 + sub_AE4AC0(a4, v20) + 24;
          v41 = *(_QWORD *)v40;
          LOBYTE(v40) = *(_BYTE *)(v40 + 8);
          v68 = v41;
          LOBYTE(v69) = v40;
          v42 = sub_CA1930(&v68);
          LODWORD(v71) = *(_DWORD *)(a5 + 8);
          if ( (unsigned int)v71 > 0x40 )
            sub_C43690(&v70, v42, 0);
          else
            v70 = v42;
          v43 = 1;
        }
        else
        {
          v57 = v36;
          v60 = v22;
          v68 = sub_BB5C00((__int64)&v64, a4);
          v69 = v47;
          v48 = sub_CA1930(&v68);
          LODWORD(v71) = *(_DWORD *)(v60 + 32);
          if ( (unsigned int)v71 > 0x40 )
            sub_C43780(&v70, v57);
          else
            v70 = *(_QWORD *)(v60 + 24);
          v43 = v48;
        }
        v44 = sub_BB50D0(v63, (__int64)&v70, v43);
        if ( (unsigned int)v71 > 0x40 && v70 )
          j_j___libc_free_0_0(v70);
        if ( !v44 )
          return v53;
        goto LABEL_32;
      }
LABEL_33:
      v30 = v19 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = v19 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v19 )
        goto LABEL_40;
      v32 = (v19 >> 1) & 3;
      if ( v32 == 2 )
      {
        if ( v30 )
          goto LABEL_36;
LABEL_40:
        v31 = sub_BCBAE0(v30, *v18);
        v18 = v64;
        goto LABEL_36;
      }
      if ( v32 != 1 || !v30 )
        goto LABEL_40;
      v31 = *(_QWORD *)(v30 + 24);
LABEL_36:
      v33 = *(_BYTE *)(v31 + 8);
      if ( v33 == 16 )
      {
        v65 = *(_QWORD *)(v31 + 24) & 0xFFFFFFFFFFFFFFF9LL | 4;
      }
      else
      {
        v34 = v31 & 0xFFFFFFFFFFFFFFF9LL;
        if ( (unsigned int)v33 - 17 > 1 )
        {
          if ( v33 != 15 )
            v34 = 0;
          v65 = v34;
        }
        else
        {
          v65 = v34 | 2;
        }
      }
      v12 = v18 + 1;
      v64 = v12;
      if ( v61 == v12 )
        return 1;
      v13 = v65;
    }
    if ( v16 == 1 && v14 )
    {
      v15 = *(_QWORD *)(v14 + 24);
      goto LABEL_10;
    }
LABEL_39:
    v15 = sub_BCBAE0(v14, *v12);
    goto LABEL_10;
  }
  v49 = (_BYTE *)*a2;
  if ( *(_BYTE *)*a2 == 17 && *(_BYTE *)(*((_QWORD *)v49 + 1) + 8LL) == 12 )
  {
    sub_C44B10(&v70, v49 + 24, *(unsigned int *)(a5 + 8));
    sub_C45EE0(a5, &v70);
    if ( (unsigned int)v71 > 0x40 )
    {
      v46 = v70;
      if ( v70 )
LABEL_75:
        j_j___libc_free_0_0(v46);
    }
  }
  else
  {
    return 0;
  }
  return v53;
}
