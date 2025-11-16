// Function: sub_EBDA90
// Address: 0xebda90
//
__int64 __fastcall sub_EBDA90(__int64 a1, unsigned __int64 a2)
{
  int *v2; // r15
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int8 v5; // al
  unsigned int i; // r14d
  __int64 *v7; // r13
  __int64 *v8; // r15
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r8
  __int64 v20; // r9
  _DWORD *v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // r14
  __int64 *v24; // r13
  unsigned __int64 v25; // rbx
  unsigned __int64 v26; // rdx
  _BOOL8 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned int v30; // eax
  unsigned __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // r12
  unsigned __int64 v34; // r13
  __int64 v35; // rbx
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // r12
  __int64 v41; // rbx
  __int64 v42; // rdi
  __int64 v43; // [rsp-8h] [rbp-258h]
  unsigned __int64 v44; // [rsp+8h] [rbp-248h]
  __int64 *v45; // [rsp+8h] [rbp-248h]
  _BOOL8 v46; // [rsp+10h] [rbp-240h]
  int *v47; // [rsp+10h] [rbp-240h]
  __int64 v48; // [rsp+28h] [rbp-228h]
  __int64 v49; // [rsp+40h] [rbp-210h]
  unsigned __int64 v50; // [rsp+48h] [rbp-208h]
  __int64 *v51; // [rsp+50h] [rbp-200h] BYREF
  __int64 *v52; // [rsp+58h] [rbp-1F8h]
  __int64 v53; // [rsp+60h] [rbp-1F0h]
  __int64 v54; // [rsp+70h] [rbp-1E0h] BYREF
  __int64 v55; // [rsp+78h] [rbp-1D8h]
  __int64 v56; // [rsp+80h] [rbp-1D0h]
  __int64 v57[2]; // [rsp+90h] [rbp-1C0h] BYREF
  __int64 v58; // [rsp+A0h] [rbp-1B0h]
  __int64 v59; // [rsp+A8h] [rbp-1A8h]
  __int64 v60; // [rsp+B0h] [rbp-1A0h]
  __int16 v61; // [rsp+B8h] [rbp-198h]
  _QWORD v62[4]; // [rsp+C0h] [rbp-190h] BYREF
  __int64 v63; // [rsp+E0h] [rbp-170h]
  __int64 v64; // [rsp+E8h] [rbp-168h]
  _QWORD *v65; // [rsp+F0h] [rbp-160h]
  _QWORD v66[3]; // [rsp+100h] [rbp-150h] BYREF
  char v67; // [rsp+118h] [rbp-138h] BYREF
  char v68; // [rsp+120h] [rbp-130h]
  char v69; // [rsp+121h] [rbp-12Fh]

  v2 = (int *)v62;
  v3 = a2;
  v4 = a1;
  v61 = 0;
  v57[0] = 0;
  v57[1] = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v62[0] = "expected identifier in '.irpc' directive";
  LOWORD(v63) = 259;
  v5 = sub_EB61F0(a1, v57);
  if ( (unsigned __int8)sub_ECE0A0(a1, v5, v62) )
    goto LABEL_2;
  v69 = 1;
  v66[0] = "expected comma";
  v68 = 3;
  if ( (unsigned __int8)sub_ECE210(a1, 26, v66) || (unsigned __int8)sub_EBC8F0(a1, 0, &v51, v16, v17, v18) )
    goto LABEL_2;
  if ( (char *)v52 - (char *)v51 != 24 || v51[1] - *v51 != 40 )
  {
    v69 = 1;
    v66[0] = "unexpected token in '.irpc' directive";
    v68 = 3;
    i = sub_ECE0E0(a1, v66, 0, 0);
    goto LABEL_3;
  }
  if ( (unsigned __int8)sub_ECE000(a1) || (v48 = sub_EB4420(a1, a2)) == 0 )
  {
LABEL_2:
    i = 1;
    goto LABEL_3;
  }
  v66[0] = &v67;
  v64 = 0x100000000LL;
  v66[1] = 0;
  v66[2] = 256;
  v62[0] = &unk_49DD288;
  v62[1] = 2;
  v62[2] = 0;
  v62[3] = 0;
  v63 = 0;
  v65 = v66;
  sub_CB5980((__int64)v62, 0, 0, 0);
  v21 = (_DWORD *)*v51;
  if ( *(_DWORD *)*v51 == 3 )
  {
    v37 = *((_QWORD *)v21 + 2);
    v38 = *((_QWORD *)v21 + 1);
    v39 = 0;
    if ( v37 )
    {
      --v37;
      v39 = 1;
      if ( !v37 )
        v37 = 1;
    }
    v50 = v37 - v39;
    v49 = v38 + v39;
  }
  else
  {
    v49 = *((_QWORD *)v21 + 1);
    v50 = *((_QWORD *)v21 + 2);
  }
  v22 = 0;
  if ( v50 )
  {
    v23 = a2;
    v24 = v57;
    v25 = 0;
    while ( 1 )
    {
      v54 = 0;
      v55 = 0;
      v56 = 0;
      if ( v25 > v50 )
      {
        v26 = v50;
        v27 = 0;
      }
      else
      {
        v26 = v25;
        v27 = v25 != v50;
      }
      v44 = v26;
      v46 = v27;
      v28 = sub_22077B0(40);
      if ( v28 )
      {
        *(_DWORD *)v28 = 2;
        *(_DWORD *)(v28 + 32) = 64;
        *(_QWORD *)(v28 + 8) = v49 + v44;
        *(_QWORD *)(v28 + 16) = v46;
        *(_QWORD *)(v28 + 24) = 0;
      }
      v29 = (__int64)v2;
      v54 = v28;
      v55 = v28 + 40;
      v56 = v28 + 40;
      v30 = sub_EA4200(v4, v2, v48, (__int64)v24, 1, 1, (__int64)&v54, 1u);
      v22 = v43;
      if ( (_BYTE)v30 )
        break;
      v19 = v54;
      if ( v55 != v54 )
      {
        v47 = v2;
        v31 = v23;
        v32 = v4;
        v33 = v55;
        v45 = v24;
        v34 = v25;
        v35 = v54;
        do
        {
          if ( *(_DWORD *)(v35 + 32) > 0x40u )
          {
            v36 = *(_QWORD *)(v35 + 24);
            if ( v36 )
              j_j___libc_free_0_0(v36);
          }
          v35 += 40;
        }
        while ( v33 != v35 );
        v25 = v34;
        v4 = v32;
        v23 = v31;
        v24 = v45;
        v2 = v47;
        v19 = v54;
      }
      if ( v19 )
        j_j___libc_free_0(v19, v56 - v19);
      if ( ++v25 == v50 )
      {
        v3 = v23;
        goto LABEL_52;
      }
    }
    v40 = v55;
    v41 = v54;
    for ( i = v30; v40 != v41; v41 += 40 )
    {
      if ( *(_DWORD *)(v41 + 32) > 0x40u )
      {
        v42 = *(_QWORD *)(v41 + 24);
        if ( v42 )
          j_j___libc_free_0_0(v42);
      }
    }
    if ( v54 )
    {
      v29 = v56 - v54;
      j_j___libc_free_0(v54, v56 - v54);
    }
  }
  else
  {
LABEL_52:
    v29 = v3;
    i = 0;
    sub_EB41F0(v4, v3, v2, v22, v19, v20);
  }
  v62[0] = &unk_49DD388;
  sub_CB5840((__int64)v2);
  if ( (char *)v66[0] != &v67 )
    _libc_free(v66[0], v29);
LABEL_3:
  v7 = v52;
  v8 = v51;
  if ( v52 != v51 )
  {
    do
    {
      v9 = v8[1];
      v10 = *v8;
      if ( v9 != *v8 )
      {
        do
        {
          if ( *(_DWORD *)(v10 + 32) > 0x40u )
          {
            v11 = *(_QWORD *)(v10 + 24);
            if ( v11 )
              j_j___libc_free_0_0(v11);
          }
          v10 += 40;
        }
        while ( v9 != v10 );
        v10 = *v8;
      }
      if ( v10 )
        j_j___libc_free_0(v10, v8[2] - v10);
      v8 += 3;
    }
    while ( v7 != v8 );
    v8 = v51;
  }
  if ( v8 )
    j_j___libc_free_0(v8, v53 - (_QWORD)v8);
  v12 = v59;
  v13 = v58;
  if ( v59 != v58 )
  {
    do
    {
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
      {
        v14 = *(_QWORD *)(v13 + 24);
        if ( v14 )
          j_j___libc_free_0_0(v14);
      }
      v13 += 40;
    }
    while ( v12 != v13 );
    v13 = v58;
  }
  if ( v13 )
    j_j___libc_free_0(v13, v60 - v13);
  return i;
}
