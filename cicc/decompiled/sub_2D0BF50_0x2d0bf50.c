// Function: sub_2D0BF50
// Address: 0x2d0bf50
//
void __fastcall sub_2D0BF50(__int64 *a1, _BYTE **a2)
{
  __int64 v3; // rdx
  __int64 **v4; // rax
  int *v6; // r8
  __int64 v7; // r9
  _BYTE *v8; // rcx
  __int64 *v9; // r13
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  __int64 v12; // rsi
  __int64 v13; // rax
  bool v14; // zf
  int *v15; // r12
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rsi
  unsigned int v26; // edi
  __int64 v27; // rax
  __int64 v28; // r9
  unsigned int v29; // ecx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int8 v33; // al
  __int64 v34; // rax
  int v35; // esi
  unsigned __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // r12
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // r12
  __int64 v52; // r13
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // r12
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 **v65; // [rsp+28h] [rbp-158h]
  int *v66; // [rsp+30h] [rbp-150h]
  int v67; // [rsp+30h] [rbp-150h]
  __int64 **v68; // [rsp+38h] [rbp-148h]
  char v69; // [rsp+4Eh] [rbp-132h] BYREF
  char v70; // [rsp+4Fh] [rbp-131h] BYREF
  __int64 *v71; // [rsp+50h] [rbp-130h] BYREF
  __int64 v72; // [rsp+58h] [rbp-128h] BYREF
  _QWORD v73[4]; // [rsp+60h] [rbp-120h] BYREF
  __int64 v74; // [rsp+80h] [rbp-100h] BYREF
  __int64 v75; // [rsp+88h] [rbp-F8h]
  __int64 v76; // [rsp+90h] [rbp-F0h]
  unsigned int v77; // [rsp+98h] [rbp-E8h]
  __int64 v78; // [rsp+A0h] [rbp-E0h] BYREF
  int v79; // [rsp+A8h] [rbp-D8h] BYREF
  unsigned __int64 v80; // [rsp+B0h] [rbp-D0h]
  int *v81; // [rsp+B8h] [rbp-C8h]
  int *v82; // [rsp+C0h] [rbp-C0h]
  __int64 v83; // [rsp+C8h] [rbp-B8h]
  int *v84[2]; // [rsp+D0h] [rbp-B0h] BYREF
  int v85; // [rsp+E0h] [rbp-A0h]
  _BYTE v86[8]; // [rsp+E8h] [rbp-98h] BYREF
  unsigned __int64 v87; // [rsp+F0h] [rbp-90h]
  char v88; // [rsp+104h] [rbp-7Ch]
  _BYTE v89[120]; // [rsp+108h] [rbp-78h] BYREF

  v81 = &v79;
  v82 = &v79;
  v73[0] = &v78;
  v3 = *((unsigned int *)a2 + 14);
  v73[1] = &v74;
  v71 = &v74;
  v4 = (__int64 **)a2[6];
  v79 = 0;
  v80 = 0;
  v83 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v73[2] = a1;
  v65 = &v4[v3];
  if ( v4 == v65 )
  {
    *((_DWORD *)a2 + 14) = 0;
    goto LABEL_70;
  }
  v68 = v4;
  while ( 2 )
  {
    v8 = *a2;
    v9 = *v68;
    v10 = **a2;
    v11 = **v68;
    if ( v10 > 0x1Cu )
    {
      if ( !(unsigned __int8)sub_2D0BA50((__int64)a1, *((_QWORD *)v8 + 5), v11, (__int64)v8) )
        goto LABEL_4;
    }
    else
    {
      if ( v10 != 22 )
        BUG();
      v12 = *(_QWORD *)(*((_QWORD *)v8 + 3) + 80LL);
      if ( v12 )
        v12 -= 24;
      if ( !(unsigned __int8)sub_2D0BA50((__int64)a1, v12, v11, (__int64)v8) )
      {
LABEL_4:
        *((_BYTE *)v9 + 184) = 1;
        goto LABEL_5;
      }
    }
    v69 = 0;
    v13 = sub_2D0A710(v73, *v9, &v69);
    if ( v13 )
    {
      v14 = v69 == 0;
      *((_BYTE *)v9 + 184) = 1;
      if ( !v14 )
        *(_BYTE *)(v13 + 185) = 0;
      goto LABEL_5;
    }
    v66 = v81;
    v15 = &v79;
    if ( v81 == &v79 )
    {
LABEL_39:
      if ( !*((_BYTE *)v9 + 184) )
        goto LABEL_40;
      goto LABEL_5;
    }
    while ( 1 )
    {
      v16 = sub_220EF80((__int64)v15);
      v17 = *(_QWORD *)(v16 + 40);
      if ( *(_BYTE *)(v17 + 184) )
      {
        v15 = (int *)v16;
        goto LABEL_38;
      }
      if ( (unsigned __int8)sub_B19720(a1[3], *(_QWORD *)v17, *v9)
        && !(unsigned __int8)sub_2D0BA50((__int64)a1, *(_QWORD *)v17, *v9, (__int64)*a2) )
      {
        break;
      }
      if ( (unsigned __int8)sub_B19720(a1[3], *v9, *(_QWORD *)v17)
        && !(unsigned __int8)sub_2D0BA50((__int64)a1, *v9, *(_QWORD *)v17, (__int64)*a2) )
      {
        sub_2D0ABA0((__int64 *)&v71, (__int64)v9, v17, v40, v41, v42);
        goto LABEL_40;
      }
      v21 = *v9;
      v22 = a1[3];
      v23 = *(_QWORD *)v17;
      v24 = *(_QWORD *)(*(_QWORD *)(*v9 + 72) + 80LL);
      if ( !v24 )
      {
        if ( !v23 )
          goto LABEL_37;
        v26 = *(_DWORD *)(v22 + 32);
        v27 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
        if ( (unsigned int)v27 >= v26 )
        {
LABEL_25:
          v28 = (unsigned int)(*(_DWORD *)(v23 + 44) + 1);
          v29 = *(_DWORD *)(v23 + 44) + 1;
          goto LABEL_26;
        }
LABEL_23:
        v24 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v27);
LABEL_24:
        if ( v23 )
          goto LABEL_25;
        v28 = 0;
        v29 = 0;
LABEL_26:
        v30 = 0;
        if ( v26 > v29 )
          v30 = *(_QWORD *)(*(_QWORD *)(v22 + 24) + 8 * v28);
        for ( ; v24 != v30; v24 = *(_QWORD *)(v24 + 8) )
        {
          if ( *(_DWORD *)(v24 + 16) < *(_DWORD *)(v30 + 16) )
          {
            v31 = v24;
            v24 = v30;
            v30 = v31;
          }
        }
        v25 = *(_QWORD *)v30;
        v72 = v25;
        if ( !v25 )
          goto LABEL_37;
        goto LABEL_33;
      }
      v25 = v24 - 24;
      if ( v21 != v25 && v23 != v25 )
      {
        v26 = *(_DWORD *)(v22 + 32);
        v24 = 0;
        v27 = (unsigned int)(*(_DWORD *)(v21 + 44) + 1);
        if ( (unsigned int)v27 < v26 )
          goto LABEL_23;
        goto LABEL_24;
      }
      v72 = v25;
LABEL_33:
      v32 = (__int64)*a2;
      v33 = **a2;
      if ( v33 <= 0x1Cu )
      {
        if ( v33 != 22 )
          BUG();
        v39 = *(_QWORD *)(*(_QWORD *)(v32 + 24) + 80LL);
        if ( !v39 )
        {
LABEL_36:
          if ( !(unsigned __int8)sub_2D0BA50((__int64)a1, v25, *(_QWORD *)v17, v32)
            && !(unsigned __int8)sub_2D0BA50((__int64)a1, v72, *v9, (__int64)*a2) )
          {
            v70 = 0;
            v43 = sub_2D0A710(v73, v72, &v70);
            v47 = v43;
            if ( v43 )
            {
              sub_2D0ABA0((__int64 *)&v71, v43, (__int64)v9, v44, v45, v46);
              sub_2D0ABA0((__int64 *)&v71, v47, v17, v48, v49, v50);
              goto LABEL_39;
            }
            v67 = *(_DWORD *)sub_2D0A0A0((__int64)(a1 + 31), &v72);
            sub_2D08230((__int64)v84, (__int64)a1, (__int64)a2, v72, 1);
            v54 = sub_22077B0(0xD0u);
            v55 = v72;
            *(_DWORD *)(v54 + 24) = v67;
            *(_QWORD *)(v54 + 16) = v55;
            *(int **)(v54 + 32) = v84[0];
            *(int **)(v54 + 40) = v84[1];
            *(_DWORD *)(v54 + 48) = v85;
            sub_C8CF70(v54 + 56, (void *)(v54 + 88), 8, (__int64)v89, (__int64)v86);
            *(_QWORD *)(v54 + 152) = v54 + 168;
            *(_QWORD *)(v54 + 160) = 0x400000000LL;
            *(_WORD *)(v54 + 200) = 0;
            sub_2208C80((_QWORD *)v54, (__int64)(a1 + 16));
            ++a1[18];
            if ( !v88 )
              _libc_free(v87);
            v59 = a1[17];
            v60 = (__int64)v9;
            *(_BYTE *)(v59 + 201) = 1;
            v61 = v59 + 16;
            v9 = (__int64 *)(v59 + 16);
            sub_2D0ABA0((__int64 *)&v71, v59 + 16, v60, v56, v57, v58);
            sub_2D0ABA0((__int64 *)&v71, v61, v17, v62, v63, v64);
            goto LABEL_40;
          }
          goto LABEL_37;
        }
        v34 = v39 - 24;
      }
      else
      {
        v34 = *(_QWORD *)(v32 + 40);
      }
      if ( v34 != v25 )
        goto LABEL_36;
LABEL_37:
      v15 = (int *)sub_220EF80((__int64)v15);
LABEL_38:
      if ( v66 == v15 )
        goto LABEL_39;
    }
    sub_2D0ABA0((__int64 *)&v71, v17, (__int64)v9, v18, v19, v20);
    if ( *((_BYTE *)v9 + 184) )
      goto LABEL_5;
LABEL_40:
    v35 = *((_DWORD *)v9 + 2);
    v36 = v80;
    v6 = &v79;
    LODWORD(v72) = v35;
    if ( !v80 )
      goto LABEL_47;
    do
    {
      while ( 1 )
      {
        v37 = *(_QWORD *)(v36 + 16);
        v38 = *(_QWORD *)(v36 + 24);
        if ( v35 <= *(_DWORD *)(v36 + 32) )
          break;
        v36 = *(_QWORD *)(v36 + 24);
        if ( !v38 )
          goto LABEL_45;
      }
      v6 = (int *)v36;
      v36 = *(_QWORD *)(v36 + 16);
    }
    while ( v37 );
LABEL_45:
    if ( v6 == &v79 || v35 < v6[8] )
    {
LABEL_47:
      v84[0] = (int *)&v72;
      v6 = (int *)sub_2D07F60(&v78, (__int64)v6, v84);
    }
    *((_QWORD *)v6 + 5) = v9;
LABEL_5:
    if ( v65 != ++v68 )
      continue;
    break;
  }
  v51 = (__int64)v81;
  for ( *((_DWORD *)a2 + 14) = 0; (int *)v51 != &v79; v51 = sub_220EEE0(v51) )
  {
    while ( 1 )
    {
      v52 = *(_QWORD *)(v51 + 40);
      if ( !*(_BYTE *)(v52 + 184) )
        break;
      v51 = sub_220EEE0(v51);
      if ( (int *)v51 == &v79 )
        goto LABEL_70;
    }
    v53 = *((unsigned int *)a2 + 14);
    if ( v53 + 1 > (unsigned __int64)*((unsigned int *)a2 + 15) )
    {
      sub_C8D5F0((__int64)(a2 + 6), a2 + 8, v53 + 1, 8u, (__int64)v6, v7);
      v53 = *((unsigned int *)a2 + 14);
    }
    *(_QWORD *)&a2[6][8 * v53] = v52;
    ++*((_DWORD *)a2 + 14);
  }
LABEL_70:
  sub_C7D6A0(v75, 16LL * v77, 8);
  sub_2D048D0(v80);
}
