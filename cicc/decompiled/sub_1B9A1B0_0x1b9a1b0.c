// Function: sub_1B9A1B0
// Address: 0x1b9a1b0
//
__int64 __fastcall sub_1B9A1B0(
        unsigned int *a1,
        unsigned __int64 a2,
        unsigned int *a3,
        __int64 a4,
        int a5,
        unsigned __int64 *a6)
{
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // rdx
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rcx
  int v14; // edx
  unsigned __int64 *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 *v17; // r14
  unsigned int v18; // r14d
  unsigned __int64 v19; // r15
  unsigned int v20; // r11d
  unsigned __int64 v21; // rax
  void *v22; // rdi
  unsigned __int64 *v23; // rax
  unsigned __int64 *v24; // r10
  __int64 v25; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // rsi
  unsigned __int64 **v28; // r11
  unsigned __int64 *v29; // r14
  unsigned __int64 v30; // r15
  unsigned __int64 *v31; // rax
  unsigned __int64 *v32; // r14
  __int64 v33; // rax
  unsigned __int64 *v34; // rcx
  unsigned __int64 *v35; // r15
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 v38; // r15
  __int64 v39; // r12
  unsigned __int64 *v40; // r9
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // rdx
  __int64 result; // rax
  unsigned __int64 v44; // rcx
  unsigned __int64 *v45; // r15
  unsigned __int64 *v46; // r14
  unsigned __int64 *v47; // rax
  __int64 v48; // rdi
  unsigned __int64 *v49; // r15
  __int64 v50; // rcx
  __int64 v51; // rsi
  __int64 v52; // rdi
  unsigned __int64 *v53; // [rsp+8h] [rbp-118h]
  unsigned __int64 *v54; // [rsp+10h] [rbp-110h]
  unsigned __int64 *v55; // [rsp+10h] [rbp-110h]
  unsigned __int64 *v56; // [rsp+18h] [rbp-108h]
  int v57; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v58; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v59; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v60; // [rsp+18h] [rbp-108h]
  unsigned __int64 *v61; // [rsp+28h] [rbp-F8h]
  int v62; // [rsp+28h] [rbp-F8h]
  unsigned __int64 *v63; // [rsp+30h] [rbp-F0h]
  unsigned int v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v66; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v67; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v68; // [rsp+30h] [rbp-F0h]
  unsigned __int64 **v69; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v70; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v71; // [rsp+30h] [rbp-F0h]
  unsigned __int64 *v72; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v74; // [rsp+48h] [rbp-D8h] BYREF
  unsigned __int64 *v75; // [rsp+50h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+58h] [rbp-C8h]
  _BYTE v77[32]; // [rsp+60h] [rbp-C0h] BYREF
  unsigned __int64 *v78; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v79; // [rsp+88h] [rbp-98h]
  _BYTE v80[144]; // [rsp+90h] [rbp-90h] BYREF

  v8 = (unsigned __int64 *)(a1 + 16);
  v9 = (unsigned __int64 *)*((_QWORD *)a1 + 9);
  v74 = a2;
  if ( !v9 )
    goto LABEL_8;
  a6 = (unsigned __int64 *)(a1 + 16);
  v10 = v9;
  do
  {
    while ( 1 )
    {
      v11 = v10[2];
      v12 = v10[3];
      if ( v10[4] >= a2 )
        break;
      v10 = (unsigned __int64 *)v10[3];
      if ( !v12 )
        goto LABEL_6;
    }
    a6 = v10;
    v10 = (unsigned __int64 *)v10[2];
  }
  while ( v11 );
LABEL_6:
  if ( v8 != a6 && a6[4] <= a2 )
  {
    v38 = a3[1];
    v39 = *a3;
  }
  else
  {
LABEL_8:
    v13 = *a1;
    v75 = (unsigned __int64 *)v77;
    v14 = v13;
    v76 = 0x400000000LL;
    v15 = (unsigned __int64 *)v80;
    v78 = (unsigned __int64 *)v80;
    v79 = 0x200000000LL;
    if ( (unsigned int)v13 > 2 )
    {
      v57 = v13;
      v68 = v13;
      sub_1B98F50(&v78, v13);
      v15 = v78;
      v14 = v57;
      v13 = v68;
    }
    LODWORD(v79) = v14;
    v16 = 48 * v13;
    v17 = &v15[6 * v13];
    if ( v17 != v15 )
    {
      do
      {
        while ( 1 )
        {
          if ( v15 )
          {
            *((_DWORD *)v15 + 2) = 0;
            *v15 = (unsigned __int64)(v15 + 2);
            *((_DWORD *)v15 + 3) = 4;
            if ( (_DWORD)v76 )
              break;
          }
          v15 += 6;
          if ( v17 == v15 )
            goto LABEL_16;
        }
        v63 = v15;
        sub_1B8E680((__int64)v15, (__int64)&v75, v16, (__int64)(v15 + 2), a5, (int)a6);
        v15 = v63 + 6;
      }
      while ( v17 != v63 + 6 );
    }
LABEL_16:
    if ( v75 != (unsigned __int64 *)v77 )
      _libc_free((unsigned __int64)v75);
    v18 = 0;
    if ( *a1 )
    {
      while ( 1 )
      {
        v19 = a1[1];
        a6 = &v78[6 * v18];
        v20 = a1[1];
        v21 = *((unsigned int *)a6 + 2);
        if ( v19 < v21 )
          goto LABEL_25;
        if ( v19 > v21 )
          break;
LABEL_26:
        if ( *a1 <= ++v18 )
          goto LABEL_27;
      }
      if ( v19 > *((unsigned int *)a6 + 3) )
      {
        v66 = &v78[6 * v18];
        sub_16CD150((__int64)v66, a6 + 2, a1[1], 8, a5, (int)a6);
        a6 = v66;
        v20 = v19;
        v21 = *((unsigned int *)v66 + 2);
      }
      v22 = (void *)(*a6 + 8 * v21);
      if ( v22 != (void *)(*a6 + 8 * v19) )
      {
        v61 = a6;
        v64 = v20;
        memset(v22, 0, 8 * (v19 - v21));
        v20 = v64;
        a6 = v61;
      }
LABEL_25:
      *((_DWORD *)a6 + 2) = v20;
      goto LABEL_26;
    }
LABEL_27:
    v23 = (unsigned __int64 *)*((_QWORD *)a1 + 9);
    v24 = v8;
    if ( !v23 )
      goto LABEL_55;
    do
    {
      while ( 1 )
      {
        v25 = v23[2];
        v26 = v23[3];
        if ( v23[4] >= v74 )
          break;
        v23 = (unsigned __int64 *)v23[3];
        if ( !v26 )
          goto LABEL_32;
      }
      v24 = v23;
      v23 = (unsigned __int64 *)v23[2];
    }
    while ( v25 );
LABEL_32:
    if ( v8 == v24 || v24[4] > v74 )
    {
LABEL_55:
      v75 = &v74;
      v24 = sub_1B99EB0((_QWORD *)a1 + 7, v24, &v75);
    }
    v27 = (unsigned int)v79;
    v28 = (unsigned __int64 **)(v24 + 5);
    v62 = v79;
    if ( v24 + 5 == (unsigned __int64 *)&v78 )
    {
      v32 = v78;
      v35 = &v78[6 * (unsigned int)v79];
    }
    else
    {
      v29 = (unsigned __int64 *)v24[5];
      v30 = *((unsigned int *)v24 + 12);
      v31 = v29;
      if ( v30 >= (unsigned int)v79 )
      {
        v44 = v24[5];
        if ( (_DWORD)v79 )
        {
          v49 = v78;
          v50 = 6LL * (unsigned int)v79;
          v60 = &v29[v50];
          do
          {
            v51 = (__int64)v49;
            v52 = (__int64)v29;
            v49 += 6;
            v29 += 6;
            v72 = v24;
            sub_1B8E680(v52, v51, v26, v50 * 8, a5, (int)a6);
            v24 = v72;
          }
          while ( v29 != v60 );
          v44 = v72[5];
          v30 = *((unsigned int *)v72 + 12);
        }
        v45 = (unsigned __int64 *)(v44 + 48 * v30);
        while ( v29 != v45 )
        {
          v45 -= 6;
          if ( (unsigned __int64 *)*v45 != v45 + 2 )
          {
            v67 = v24;
            _libc_free(*v45);
            v24 = v67;
          }
        }
        *((_DWORD *)v24 + 12) = v62;
        v32 = v78;
        v35 = &v78[6 * (unsigned int)v79];
      }
      else
      {
        if ( *((_DWORD *)v24 + 13) < (unsigned int)v79 )
        {
          v46 = &v29[6 * v30];
          while ( v46 != v31 )
          {
            while ( 1 )
            {
              v46 -= 6;
              if ( (unsigned __int64 *)*v46 == v46 + 2 )
                break;
              v53 = v31;
              v58 = v24;
              v69 = v28;
              _libc_free(*v46);
              v31 = v53;
              v24 = v58;
              v28 = v69;
              if ( v46 == v53 )
                goto LABEL_77;
            }
          }
LABEL_77:
          *((_DWORD *)v24 + 12) = 0;
          v30 = 0;
          v70 = v24;
          sub_1B98F50(v28, v27);
          v24 = v70;
          v27 = (unsigned int)v79;
          v31 = (unsigned __int64 *)v70[5];
        }
        else if ( *((_DWORD *)v24 + 12) )
        {
          LODWORD(a6) = 3 * v30;
          v47 = v78;
          v30 *= 48LL;
          v55 = (unsigned __int64 *)((char *)v29 + v30);
          do
          {
            v48 = (__int64)v29;
            v59 = v24;
            v29 += 6;
            v71 = v47;
            sub_1B8E680(v48, (__int64)v47, v26, v25, a5, (int)a6);
            v24 = v59;
            v47 = v71 + 6;
          }
          while ( v29 != v55 );
          v27 = (unsigned int)v79;
          v31 = (unsigned __int64 *)v59[5];
        }
        v32 = v78;
        v33 = (__int64)v31 + v30;
        v34 = &v78[6 * v27];
        v35 = (unsigned __int64 *)((char *)v78 + v30);
        if ( v34 != v35 )
        {
          do
          {
            while ( 1 )
            {
              if ( v33 )
              {
                *(_DWORD *)(v33 + 8) = 0;
                *(_QWORD *)v33 = v33 + 16;
                *(_DWORD *)(v33 + 12) = 4;
                v36 = *((unsigned int *)v35 + 2);
                if ( (_DWORD)v36 )
                  break;
              }
              v35 += 6;
              v33 += 48;
              if ( v34 == v35 )
                goto LABEL_44;
            }
            v37 = (__int64)v35;
            v54 = v34;
            v35 += 6;
            v56 = v24;
            v65 = v33;
            sub_1B8E680(v33, v37, v36, (__int64)v34, a5, (int)a6);
            v34 = v54;
            v24 = v56;
            v33 = v65 + 48;
          }
          while ( v54 != v35 );
LABEL_44:
          v32 = v78;
          v35 = &v78[6 * (unsigned int)v79];
        }
        *((_DWORD *)v24 + 12) = v62;
      }
    }
    if ( v32 != v35 )
    {
      do
      {
        v35 -= 6;
        if ( (unsigned __int64 *)*v35 != v35 + 2 )
          _libc_free(*v35);
      }
      while ( v35 != v32 );
      v35 = v78;
    }
    if ( v35 != (unsigned __int64 *)v80 )
      _libc_free((unsigned __int64)v35);
    v9 = (unsigned __int64 *)*((_QWORD *)a1 + 9);
    v38 = a3[1];
    v39 = *a3;
    if ( !v9 )
    {
      v40 = v8;
LABEL_53:
      v78 = &v74;
      v40 = sub_1B99EB0((_QWORD *)a1 + 7, v40, &v78);
      goto LABEL_54;
    }
    a2 = v74;
  }
  v40 = v8;
  do
  {
    while ( 1 )
    {
      v41 = v9[2];
      v42 = v9[3];
      if ( v9[4] >= a2 )
        break;
      v9 = (unsigned __int64 *)v9[3];
      if ( !v42 )
        goto LABEL_51;
    }
    v40 = v9;
    v9 = (unsigned __int64 *)v9[2];
  }
  while ( v41 );
LABEL_51:
  if ( v8 == v40 || v40[4] > a2 )
    goto LABEL_53;
LABEL_54:
  result = *(_QWORD *)(v40[5] + 48 * v39);
  *(_QWORD *)(result + 8 * v38) = a4;
  return result;
}
