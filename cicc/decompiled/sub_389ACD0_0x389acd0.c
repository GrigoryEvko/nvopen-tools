// Function: sub_389ACD0
// Address: 0x389acd0
//
__int64 *__fastcall sub_389ACD0(__int64 *a1, unsigned int a2, __int64 a3, unsigned __int64 a4, char a5)
{
  __int64 v9; // rdx
  __int64 *v10; // r9
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // r12
  __int64 v14; // rax
  _QWORD *v15; // r14
  _QWORD *v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rdi
  __int64 *v22; // r14
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r15
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // r9
  char v34; // di
  __int64 v35; // rcx
  __int64 v36; // r15
  __int64 *v37; // rax
  __int64 v38; // rcx
  __int64 *v39; // [rsp+0h] [rbp-120h]
  unsigned __int64 v40; // [rsp+0h] [rbp-120h]
  __int64 *v41; // [rsp+8h] [rbp-118h]
  __int64 *v42; // [rsp+8h] [rbp-118h]
  __int64 v43; // [rsp+8h] [rbp-118h]
  __int64 v44; // [rsp+8h] [rbp-118h]
  __int64 *v45; // [rsp+8h] [rbp-118h]
  __int64 *v46; // [rsp+8h] [rbp-118h]
  __int64 v47; // [rsp+10h] [rbp-110h]
  _QWORD v48[2]; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v49; // [rsp+40h] [rbp-E0h]
  _QWORD v50[2]; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v51; // [rsp+60h] [rbp-C0h]
  _QWORD v52[2]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+80h] [rbp-A0h]
  char *v54; // [rsp+90h] [rbp-90h]
  char v55; // [rsp+A0h] [rbp-80h]
  char v56; // [rsp+A1h] [rbp-7Fh]
  char *v57; // [rsp+B0h] [rbp-70h] BYREF
  char *v58; // [rsp+B8h] [rbp-68h]
  __int64 v59; // [rsp+C0h] [rbp-60h]
  __int64 v60[2]; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v61; // [rsp+E0h] [rbp-40h] BYREF

  v9 = a1[14];
  if ( a2 < (unsigned __int64)((a1[15] - v9) >> 3) && (v10 = *(__int64 **)(v9 + 8LL * a2)) != 0 )
  {
    if ( a3 != *v10 )
    {
LABEL_4:
      v11 = *a1;
      v12 = *(_BYTE *)(a3 + 8);
      if ( a5 && v12 == 15 )
      {
        v22 = *(__int64 **)(a3 + 24);
        v41 = v10;
        v23 = sub_1632FA0(*(_QWORD *)(v11 + 176));
        v24 = sub_1647190(v22, *(_DWORD *)(v23 + 12));
        v10 = v41;
        if ( *v41 == v24 )
          return v10;
        v11 = *a1;
        v12 = *(_BYTE *)(a3 + 8);
      }
      v13 = v11 + 8;
      if ( v12 == 7 )
      {
        LODWORD(v54) = a2;
        v57 = "'%";
        LOWORD(v59) = 2307;
        v58 = v54;
        v60[0] = (__int64)&v57;
        v60[1] = (__int64)"' is not a basic block";
        LOWORD(v61) = 770;
        sub_38814C0(v13, a4, (__int64)v60);
        return 0;
      }
      v56 = 1;
      v54 = "'";
      v55 = 3;
      sub_3888960(v60, *v10);
      LODWORD(v47) = a2;
      v48[0] = "'%";
      v49 = 2307;
      v48[1] = v47;
      v50[0] = v48;
      v50[1] = "' defined with type '";
      v52[0] = v50;
      v51 = 770;
      v52[1] = v60;
      LOWORD(v53) = 1026;
      v58 = "'";
      v57 = (char *)v52;
      LOWORD(v59) = 770;
      sub_38814C0(v13, a4, (__int64)&v57);
      if ( (__int64 *)v60[0] != &v61 )
        j_j___libc_free_0(v60[0]);
      return 0;
    }
  }
  else
  {
    v14 = a1[10];
    v15 = a1 + 9;
    if ( v14 )
    {
      v16 = a1 + 9;
      do
      {
        while ( 1 )
        {
          v17 = *(_QWORD *)(v14 + 16);
          v18 = *(_QWORD *)(v14 + 24);
          if ( *(_DWORD *)(v14 + 32) >= a2 )
            break;
          v14 = *(_QWORD *)(v14 + 24);
          if ( !v18 )
            goto LABEL_12;
        }
        v16 = (_QWORD *)v14;
        v14 = *(_QWORD *)(v14 + 16);
      }
      while ( v17 );
LABEL_12:
      if ( v15 != v16 && *((_DWORD *)v16 + 8) <= a2 )
      {
        v10 = (__int64 *)v16[5];
        if ( v10 )
        {
          if ( a3 == *v10 )
            return v10;
          goto LABEL_4;
        }
      }
    }
    v19 = *(unsigned __int8 *)(a3 + 8);
    if ( !*(_BYTE *)(a3 + 8) || v19 == 12 )
    {
      v20 = *a1;
      LOWORD(v61) = 259;
      v60[0] = (__int64)"invalid use of a non-first-class type";
      sub_38814C0(v20 + 8, a4, (__int64)v60);
      return 0;
    }
    if ( (_BYTE)v19 == 7 )
    {
      v35 = a1[1];
      LOWORD(v61) = 257;
      v44 = v35;
      v36 = sub_15E0530(v35);
      v37 = (__int64 *)sub_22077B0(0x40u);
      v10 = v37;
      if ( v37 )
      {
        v38 = v44;
        v45 = v37;
        sub_157FB60(v37, v36, (__int64)v60, v38, 0);
        v10 = v45;
      }
    }
    else
    {
      LOWORD(v61) = 257;
      v25 = sub_22077B0(0x28u);
      v10 = (__int64 *)v25;
      if ( v25 )
      {
        v42 = (__int64 *)v25;
        sub_15E0280(v25, a3, (__int64)v60, 0, 0);
        v10 = v42;
      }
    }
    v26 = a1[10];
    v27 = (unsigned __int64)(a1 + 9);
    if ( !v26 )
      goto LABEL_35;
    do
    {
      while ( 1 )
      {
        v28 = *(_QWORD *)(v26 + 16);
        v29 = *(_QWORD *)(v26 + 24);
        if ( *(_DWORD *)(v26 + 32) >= a2 )
          break;
        v26 = *(_QWORD *)(v26 + 24);
        if ( !v29 )
          goto LABEL_33;
      }
      v27 = v26;
      v26 = *(_QWORD *)(v26 + 16);
    }
    while ( v28 );
LABEL_33:
    if ( v15 == (_QWORD *)v27 || *(_DWORD *)(v27 + 32) > a2 )
    {
LABEL_35:
      v39 = v10;
      v43 = v27;
      v30 = sub_22077B0(0x38u);
      *(_DWORD *)(v30 + 32) = a2;
      v27 = v30;
      *(_QWORD *)(v30 + 40) = 0;
      *(_QWORD *)(v30 + 48) = 0;
      v31 = sub_389ABD0(a1 + 8, v43, (unsigned int *)(v30 + 32));
      v33 = v39;
      if ( v32 )
      {
        v34 = v15 == (_QWORD *)v32 || v31 || a2 < *(_DWORD *)(v32 + 32);
        sub_220F040(v34, v27, (_QWORD *)v32, a1 + 9);
        ++a1[13];
        v10 = v39;
      }
      else
      {
        v40 = v31;
        v46 = v33;
        j_j___libc_free_0(v27);
        v10 = v46;
        v27 = v40;
      }
    }
    *(_QWORD *)(v27 + 40) = v10;
    *(_QWORD *)(v27 + 48) = a4;
  }
  return v10;
}
