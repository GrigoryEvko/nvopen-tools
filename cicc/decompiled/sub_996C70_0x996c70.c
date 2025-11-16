// Function: sub_996C70
// Address: 0x996c70
//
char __fastcall sub_996C70(__int64 a1, __int64 a2, _QWORD *a3, unsigned int a4, __int64 *a5, char a6)
{
  unsigned int *v9; // rax
  unsigned int v10; // edx
  unsigned __int8 v11; // al
  unsigned int *v12; // rax
  unsigned int v13; // eax
  __int64 v14; // rdi
  bool v15; // zf
  unsigned __int8 v16; // al
  __int64 v17; // rdi
  _BYTE *v18; // rdi
  char v19; // al
  __int64 v20; // r10
  unsigned int v21; // ebx
  __int64 v22; // rdi
  unsigned int v23; // r13d
  bool v24; // cc
  int v25; // eax
  __int64 v26; // rdi
  __int64 *v27; // rdx
  char v28; // al
  __int64 *v29; // rax
  __int64 v30; // rdi
  _BYTE *v31; // rdi
  _BYTE *v32; // rdi
  __int64 v34; // [rsp+8h] [rbp-E8h]
  __int64 v35; // [rsp+10h] [rbp-E0h]
  __int64 v36; // [rsp+10h] [rbp-E0h]
  __int64 v37; // [rsp+18h] [rbp-D8h]
  __int64 v38; // [rsp+18h] [rbp-D8h]
  __int64 v41; // [rsp+38h] [rbp-B8h] BYREF
  unsigned __int64 v42; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v43; // [rsp+48h] [rbp-A8h]
  _QWORD *v44; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+58h] [rbp-98h]
  __int64 v46; // [rsp+60h] [rbp-90h] BYREF
  __int64 v47; // [rsp+68h] [rbp-88h]
  __int64 v48; // [rsp+70h] [rbp-80h] BYREF
  __int64 v49; // [rsp+78h] [rbp-78h]
  _QWORD *v50; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v51; // [rsp+88h] [rbp-68h]
  _QWORD *v52; // [rsp+90h] [rbp-60h]
  unsigned int v53; // [rsp+98h] [rbp-58h]
  _QWORD *v54; // [rsp+A0h] [rbp-50h] BYREF
  __int64 *v55; // [rsp+A8h] [rbp-48h]
  __int64 v56; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v57; // [rsp+B8h] [rbp-38h]

  v9 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v9 )
    v10 = *v9;
  else
    v10 = qword_4F862D0[2];
  v11 = *(_BYTE *)a2;
  if ( a4 >= v10 )
    goto LABEL_4;
  if ( v11 <= 0x1Cu )
    goto LABEL_7;
  v14 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  v15 = (unsigned __int8)sub_BCAC40(v14, 1) == 0;
  v16 = *(_BYTE *)a2;
  if ( v15 )
    goto LABEL_91;
  if ( v16 == 57 )
  {
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v27 = *(__int64 **)(a2 - 8);
    else
      v27 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( *v27 )
    {
      v20 = v27[4];
      v41 = *v27;
      if ( v20 )
        goto LABEL_27;
    }
    goto LABEL_92;
  }
  if ( v16 != 86 )
  {
LABEL_91:
    if ( v16 <= 0x1Cu )
      goto LABEL_7;
LABEL_92:
    v17 = *(_QWORD *)(a2 + 8);
    goto LABEL_18;
  }
  v17 = *(_QWORD *)(a2 + 8);
  v37 = *(_QWORD *)(a2 - 96);
  if ( *(_QWORD *)(v37 + 8) == v17 && **(_BYTE **)(a2 - 32) <= 0x15u )
  {
    v36 = *(_QWORD *)(a2 - 64);
    v28 = sub_AC30F0(*(_QWORD *)(a2 - 32));
    v20 = v36;
    if ( v28 )
    {
      v41 = v37;
      if ( v36 )
        goto LABEL_27;
    }
    v16 = *(_BYTE *)a2;
    goto LABEL_91;
  }
LABEL_18:
  if ( (unsigned int)*(unsigned __int8 *)(v17 + 8) - 17 <= 1 )
    v17 = **(_QWORD **)(v17 + 16);
  v15 = (unsigned __int8)sub_BCAC40(v17, 1) == 0;
  v11 = *(_BYTE *)a2;
  if ( v15 )
    goto LABEL_4;
  if ( v11 == 58 )
  {
    v29 = (__int64 *)sub_986520(a2);
    if ( !*v29 )
      goto LABEL_7;
    v20 = v29[4];
    v41 = *v29;
    if ( !v20 )
      goto LABEL_7;
    goto LABEL_27;
  }
  if ( v11 == 86 )
  {
    v38 = *(_QWORD *)(a2 - 96);
    if ( *(_QWORD *)(v38 + 8) != *(_QWORD *)(a2 + 8) )
      goto LABEL_7;
    v18 = *(_BYTE **)(a2 - 64);
    if ( *v18 > 0x15u )
      goto LABEL_7;
    v35 = *(_QWORD *)(a2 - 32);
    v19 = sub_AD7A80(v18);
    v20 = v35;
    if ( v19 )
    {
      v41 = v38;
      if ( v35 )
      {
LABEL_27:
        v34 = v20;
        v21 = a4 + 1;
        sub_9878D0((__int64)&v46, *((_DWORD *)a3 + 2));
        sub_9878D0((__int64)&v50, *((_DWORD *)a3 + 2));
        sub_996C70(a1, v41, &v46, v21, a5);
        sub_996C70(a1, v34, &v50, v21, a5);
        if ( a6 )
        {
          if ( *(_BYTE *)a2 <= 0x1Cu )
            goto LABEL_35;
          v22 = *(_QWORD *)(a2 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 <= 1 )
            v22 = **(_QWORD **)(v22 + 16);
          if ( !(unsigned __int8)sub_BCAC40(v22, 1)
            || *(_BYTE *)a2 != 58
            && (*(_BYTE *)a2 != 86
             || *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) != *(_QWORD *)(a2 + 8)
             || (v32 = *(_BYTE **)(a2 - 64), *v32 > 0x15u)
             || !(unsigned __int8)sub_AD7A80(v32)) )
          {
LABEL_35:
            sub_987C60((__int64)&v54, &v46, &v50);
LABEL_36:
            sub_984AC0(&v46, (__int64 *)&v54);
            sub_969240(&v56);
            sub_969240((__int64 *)&v54);
            sub_987D70((__int64)&v54, a3, &v46);
            LOBYTE(v13) = sub_984AC0(a3, (__int64 *)&v54);
            if ( v57 > 0x40 && v56 )
              LOBYTE(v13) = j_j___libc_free_0_0(v56);
            goto LABEL_56;
          }
        }
        else
        {
          if ( *(_BYTE *)a2 <= 0x1Cu )
            goto LABEL_35;
          v30 = *(_QWORD *)(a2 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17 <= 1 )
            v30 = **(_QWORD **)(v30 + 16);
          if ( !(unsigned __int8)sub_BCAC40(v30, 1) )
            goto LABEL_35;
          if ( *(_BYTE *)a2 != 57 )
          {
            if ( *(_BYTE *)a2 != 86 )
              goto LABEL_35;
            if ( *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) != *(_QWORD *)(a2 + 8) )
              goto LABEL_35;
            v31 = *(_BYTE **)(a2 - 32);
            if ( *v31 > 0x15u || !(unsigned __int8)sub_AC30F0(v31) )
              goto LABEL_35;
          }
        }
        sub_987D70((__int64)&v54, &v46, &v50);
        goto LABEL_36;
      }
    }
    v11 = *(_BYTE *)a2;
  }
LABEL_4:
  if ( v11 > 0x1Cu )
  {
    if ( v11 == 82 )
    {
      LOBYTE(v13) = sub_993630(a1, a2, a3, a5, a6);
      return v13;
    }
    if ( v11 == 67 && a1 == *(_QWORD *)(a2 - 32) )
    {
      v46 = 0;
      v47 = 1;
      v48 = 0;
      v49 = 1;
      if ( a6 )
        v46 = 1;
      else
        v48 = 1;
      v23 = *((_DWORD *)a3 + 2);
      if ( (*(_BYTE *)(a2 + 1) & 2) != 0 )
      {
        sub_C449B0(&v42, &v46, v23);
        if ( v43 != 1 )
        {
          if ( v43 > 0x40 )
            sub_C43C90(&v42, 1, v43);
          else
            v42 |= 2 * (0xFFFFFFFFFFFFFFFFLL >> (65 - (unsigned __int8)v43));
        }
        sub_C449B0(&v44, &v48, v23);
        LODWORD(v55) = v43;
        if ( v43 > 0x40 )
        {
          sub_C43780(&v54, &v42);
          v51 = (unsigned int)v55;
          v50 = v54;
          v53 = v45;
          v52 = v44;
          if ( v43 > 0x40 && v42 )
            j_j___libc_free_0_0(v42);
        }
        else
        {
          v51 = v43;
          v50 = (_QWORD *)v42;
          v53 = v45;
          v52 = v44;
        }
        sub_987D70((__int64)&v54, a3, &v50);
        if ( *((_DWORD *)a3 + 2) <= 0x40u )
          goto LABEL_53;
      }
      else
      {
        sub_C449B0(&v54, &v48, v23);
        sub_C449B0(&v44, &v46, v23);
        v51 = v45;
        v50 = v44;
        v53 = (unsigned int)v55;
        v52 = v54;
        sub_987D70((__int64)&v54, a3, &v50);
        if ( *((_DWORD *)a3 + 2) <= 0x40u )
        {
LABEL_53:
          v24 = *((_DWORD *)a3 + 6) <= 0x40u;
          *a3 = v54;
          v25 = (int)v55;
          LODWORD(v55) = 0;
          *((_DWORD *)a3 + 2) = v25;
          if ( v24 || (v26 = a3[2]) == 0 )
          {
            a3[2] = v56;
            LOBYTE(v13) = v57;
            *((_DWORD *)a3 + 6) = v57;
LABEL_59:
            if ( v53 > 0x40 && v52 )
              LOBYTE(v13) = j_j___libc_free_0_0(v52);
            if ( v51 > 0x40 && v50 )
              LOBYTE(v13) = j_j___libc_free_0_0(v50);
            if ( (unsigned int)v49 > 0x40 && v48 )
              LOBYTE(v13) = j_j___libc_free_0_0(v48);
            if ( (unsigned int)v47 > 0x40 && v46 )
              LOBYTE(v13) = j_j___libc_free_0_0(v46);
            return v13;
          }
          j_j___libc_free_0_0(v26);
          a3[2] = v56;
          LOBYTE(v13) = v57;
          *((_DWORD *)a3 + 6) = v57;
LABEL_56:
          if ( (unsigned int)v55 > 0x40 && v54 )
            LOBYTE(v13) = j_j___libc_free_0_0(v54);
          goto LABEL_59;
        }
      }
      if ( *a3 )
        j_j___libc_free_0_0(*a3);
      goto LABEL_53;
    }
  }
LABEL_7:
  v12 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v12 )
    v13 = *v12;
  else
    v13 = qword_4F862D0[2];
  if ( a4 < v13 )
  {
    v54 = 0;
    v55 = &v41;
    LOBYTE(v13) = sub_996420(&v54, 30, (unsigned __int8 *)a2);
    if ( (_BYTE)v13 )
      LOBYTE(v13) = sub_996C70(a1, v41, a3, a4 + 1, a5);
  }
  return v13;
}
