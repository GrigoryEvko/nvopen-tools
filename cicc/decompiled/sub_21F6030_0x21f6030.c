// Function: sub_21F6030
// Address: 0x21f6030
//
__int64 __fastcall sub_21F6030(__int64 a1, unsigned __int64 a2, _QWORD *a3)
{
  unsigned __int64 **v4; // r13
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // rcx
  int *v7; // r14
  unsigned __int64 *v8; // r13
  unsigned __int64 v9; // rdx
  int *v10; // rax
  __int64 v11; // r15
  unsigned __int64 v12; // r14
  unsigned __int8 v13; // al
  _QWORD *v14; // r13
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  _BOOL4 v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdi
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 v22; // rdi
  __int64 v24; // rax
  _BOOL4 v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r13
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rcx
  int *v31; // r15
  unsigned __int64 v32; // rdx
  int *v33; // rax
  _BOOL4 v34; // r10d
  __int64 v35; // rax
  unsigned __int64 *v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rcx
  int *v39; // r13
  int *i; // r15
  unsigned __int64 v41; // rdx
  int *v42; // rax
  _BOOL4 v43; // r13d
  __int64 v44; // rax
  unsigned __int64 *v45; // rax
  unsigned __int64 v46; // r15
  unsigned __int64 v47; // rdx
  int *v48; // rax
  _BOOL4 v49; // r14d
  __int64 v50; // rax
  unsigned __int64 *v51; // r13
  __int64 v52; // rax
  _QWORD *v53; // [rsp+8h] [rbp-D8h]
  _BOOL4 v54; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v55; // [rsp+10h] [rbp-D0h]
  unsigned __int64 *v56; // [rsp+18h] [rbp-C8h]
  __int64 v57; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v58; // [rsp+18h] [rbp-C8h]
  _QWORD v59[2]; // [rsp+28h] [rbp-B8h] BYREF
  int v60; // [rsp+38h] [rbp-A8h] BYREF
  int *v61; // [rsp+40h] [rbp-A0h]
  int *v62; // [rsp+48h] [rbp-98h]
  int *v63; // [rsp+50h] [rbp-90h]
  __int64 v64; // [rsp+58h] [rbp-88h]
  __int64 v65; // [rsp+60h] [rbp-80h] BYREF
  __int64 v66; // [rsp+68h] [rbp-78h]
  unsigned __int64 *v67; // [rsp+70h] [rbp-70h]
  unsigned __int64 *v68; // [rsp+78h] [rbp-68h]
  _QWORD *v69; // [rsp+80h] [rbp-60h]
  unsigned __int64 v70; // [rsp+88h] [rbp-58h]
  unsigned __int64 *v71; // [rsp+90h] [rbp-50h]
  unsigned __int64 *v72; // [rsp+98h] [rbp-48h]
  _QWORD *v73; // [rsp+A0h] [rbp-40h]
  unsigned __int64 **v74; // [rsp+A8h] [rbp-38h]

  v62 = &v60;
  v60 = 0;
  v61 = 0;
  v63 = &v60;
  v64 = 0;
  v65 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v66 = 8;
  v65 = sub_22077B0(64);
  v4 = (unsigned __int64 **)(v65 + ((4 * v66 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v5 = (unsigned __int64 *)sub_22077B0(512);
  v70 = (unsigned __int64)v4;
  v6 = v5;
  *v4 = v5;
  v68 = v5;
  v69 = v5 + 64;
  v74 = v4;
  v72 = v5;
  v73 = v5 + 64;
  v67 = v5;
  if ( v5 )
    *v5 = a2;
  v7 = v61;
  v8 = v5 + 1;
  v71 = v5 + 1;
  if ( v61 )
  {
    while ( 1 )
    {
      v9 = *((_QWORD *)v7 + 4);
      v10 = (int *)*((_QWORD *)v7 + 3);
      if ( a2 < v9 )
        v10 = (int *)*((_QWORD *)v7 + 2);
      if ( !v10 )
        break;
      v7 = v10;
    }
    if ( a2 < v9 )
    {
      if ( v62 != v7 )
        goto LABEL_36;
    }
    else if ( v9 >= a2 )
    {
      goto LABEL_11;
    }
LABEL_37:
    v25 = 1;
    if ( v7 != &v60 )
      v25 = a2 < *((_QWORD *)v7 + 4);
    goto LABEL_39;
  }
  v7 = &v60;
  if ( v62 != &v60 )
  {
LABEL_36:
    v56 = v6;
    v24 = sub_220EF80(v7);
    v6 = v56;
    if ( *(_QWORD *)(v24 + 32) >= a2 )
      goto LABEL_11;
    goto LABEL_37;
  }
  v25 = 1;
LABEL_39:
  v26 = sub_22077B0(40);
  *(_QWORD *)(v26 + 32) = a2;
  sub_220F040(v25, v26, v7, &v60);
  v8 = v71;
  v6 = v67;
  ++v64;
LABEL_11:
  v53 = a3 + 1;
  if ( v8 != v6 )
  {
    while ( 1 )
    {
      if ( v72 == v8 )
      {
        v11 = (*(v74 - 1))[63];
        j_j___libc_free_0(v8, 512);
        v27 = (__int64)(*--v74 + 64);
        v72 = *v74;
        v73 = (_QWORD *)v27;
        v71 = v72 + 63;
      }
      else
      {
        v11 = *(v8 - 1);
        v71 = v8 - 1;
      }
      v12 = sub_1CCAE90(v11, 1);
      v13 = *(_BYTE *)(v12 + 16);
      if ( v13 > 0x17u )
      {
        if ( v13 == 77 )
        {
          if ( (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) != 0 )
          {
            v28 = 0;
            v57 = 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
            if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
            {
LABEL_47:
              v29 = *(_QWORD *)(v12 - 8);
              goto LABEL_48;
            }
            while ( 2 )
            {
              v29 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
LABEL_48:
              v30 = *(_QWORD *)(v29 + v28);
              v31 = v61;
              v59[0] = v30;
              if ( !v61 )
              {
                v31 = &v60;
                if ( v62 == &v60 )
                {
                  v34 = 1;
                  goto LABEL_57;
                }
LABEL_65:
                v55 = v30;
                v37 = sub_220EF80(v31);
                v30 = v55;
                if ( *(_QWORD *)(v37 + 32) < v55 )
                {
                  v34 = 1;
                  if ( v31 != &v60 )
LABEL_67:
                    v34 = v30 < *((_QWORD *)v31 + 4);
LABEL_57:
                  v54 = v34;
                  v35 = sub_22077B0(40);
                  *(_QWORD *)(v35 + 32) = v59[0];
                  sub_220F040(v54, v35, v31, &v60);
                  v36 = v71;
                  ++v64;
                  if ( v71 == v73 - 1 )
                  {
                    sub_21F5E30(&v65, v59);
                  }
                  else
                  {
                    if ( v71 )
                    {
                      *v71 = v59[0];
                      v36 = v71;
                    }
                    v71 = v36 + 1;
                  }
                }
LABEL_61:
                v28 += 24;
                if ( v57 == v28 )
                  goto LABEL_27;
                if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
                  goto LABEL_47;
                continue;
              }
              break;
            }
            while ( 1 )
            {
              v32 = *((_QWORD *)v31 + 4);
              v33 = (int *)*((_QWORD *)v31 + 3);
              if ( v30 < v32 )
                v33 = (int *)*((_QWORD *)v31 + 2);
              if ( !v33 )
                break;
              v31 = v33;
            }
            if ( v30 < v32 )
            {
              if ( v62 != v31 )
                goto LABEL_65;
            }
            else if ( v30 <= v32 )
            {
              goto LABEL_61;
            }
            v34 = 1;
            if ( v31 != &v60 )
              goto LABEL_67;
            goto LABEL_57;
          }
          goto LABEL_27;
        }
        if ( v13 == 79 )
          break;
      }
      v14 = (_QWORD *)a3[2];
      if ( !v14 )
      {
        v14 = a3 + 1;
        if ( v53 == (_QWORD *)a3[3] )
        {
          v14 = a3 + 1;
          v17 = 1;
          goto LABEL_26;
        }
LABEL_42:
        if ( v12 <= *(_QWORD *)(sub_220EF80(v14) + 32) )
          goto LABEL_27;
        v17 = 1;
        if ( v14 == v53 )
          goto LABEL_26;
        goto LABEL_44;
      }
      while ( 1 )
      {
        v15 = v14[4];
        v16 = (_QWORD *)v14[3];
        if ( v12 < v15 )
          v16 = (_QWORD *)v14[2];
        if ( !v16 )
          break;
        v14 = v16;
      }
      if ( v12 < v15 )
      {
        if ( (_QWORD *)a3[3] != v14 )
          goto LABEL_42;
LABEL_25:
        v17 = 1;
        if ( v14 == v53 )
        {
LABEL_26:
          v18 = sub_22077B0(40);
          *(_QWORD *)(v18 + 32) = v12;
          sub_220F040(v17, v18, v14, v53);
          ++a3[5];
          goto LABEL_27;
        }
LABEL_44:
        v17 = v12 < v14[4];
        goto LABEL_26;
      }
      if ( v12 > v15 )
        goto LABEL_25;
LABEL_27:
      v8 = v71;
LABEL_28:
      if ( v67 == v8 )
        goto LABEL_29;
    }
    v38 = *(_QWORD *)(v12 - 48);
    v39 = v61;
    v59[0] = v38;
    if ( v61 )
    {
      for ( i = v61; ; i = v42 )
      {
        v41 = *((_QWORD *)i + 4);
        v42 = (int *)*((_QWORD *)i + 3);
        if ( v38 < v41 )
          v42 = (int *)*((_QWORD *)i + 2);
        if ( !v42 )
          break;
      }
      if ( v38 >= v41 )
      {
        if ( v38 <= v41 )
          goto LABEL_84;
        goto LABEL_78;
      }
      if ( v62 == i )
      {
LABEL_78:
        v43 = 1;
        if ( i != &v60 )
          v43 = v38 < *((_QWORD *)i + 4);
        goto LABEL_80;
      }
    }
    else
    {
      i = &v60;
      if ( v62 == &v60 )
      {
        v43 = 1;
LABEL_80:
        v44 = sub_22077B0(40);
        *(_QWORD *)(v44 + 32) = v59[0];
        sub_220F040(v43, v44, i, &v60);
        v45 = v71;
        ++v64;
        if ( v71 == v73 - 1 )
        {
          sub_21F5E30(&v65, v59);
          v39 = v61;
        }
        else
        {
          if ( v71 )
          {
            *v71 = v59[0];
            v45 = v71;
          }
          v39 = v61;
          v71 = v45 + 1;
        }
LABEL_84:
        v46 = *(_QWORD *)(v12 - 24);
        v59[0] = v46;
        if ( !v39 )
        {
LABEL_107:
          v39 = &v60;
          if ( v62 == &v60 )
          {
            v49 = 1;
            goto LABEL_93;
          }
LABEL_101:
          if ( *(_QWORD *)(sub_220EF80(v39) + 32) < v46 )
          {
            v49 = 1;
            if ( v39 == &v60 )
              goto LABEL_93;
            goto LABEL_109;
          }
          goto LABEL_27;
        }
        while ( 1 )
        {
LABEL_87:
          v47 = *((_QWORD *)v39 + 4);
          v48 = (int *)*((_QWORD *)v39 + 3);
          if ( v46 < v47 )
            v48 = (int *)*((_QWORD *)v39 + 2);
          if ( !v48 )
            break;
          v39 = v48;
        }
        if ( v46 >= v47 )
        {
          if ( v47 < v46 )
            goto LABEL_92;
          goto LABEL_27;
        }
        if ( v62 == v39 )
        {
LABEL_92:
          v49 = 1;
          if ( v39 == &v60 )
            goto LABEL_93;
LABEL_109:
          v49 = v46 < *((_QWORD *)v39 + 4);
LABEL_93:
          v50 = sub_22077B0(40);
          *(_QWORD *)(v50 + 32) = v59[0];
          sub_220F040(v49, v50, v39, &v60);
          v51 = v71;
          ++v64;
          if ( v71 == v73 - 1 )
          {
            sub_21F5E30(&v65, v59);
            v8 = v71;
          }
          else
          {
            if ( v71 )
            {
              *v71 = v59[0];
              v51 = v71;
            }
            v8 = v51 + 1;
            v71 = v8;
          }
          goto LABEL_28;
        }
        goto LABEL_101;
      }
    }
    v58 = v38;
    v52 = sub_220EF80(i);
    v38 = v58;
    if ( v58 <= *(_QWORD *)(v52 + 32) )
    {
      v46 = *(_QWORD *)(v12 - 24);
      v59[0] = v46;
      if ( !v39 )
        goto LABEL_107;
      goto LABEL_87;
    }
    goto LABEL_78;
  }
LABEL_29:
  v19 = v65;
  if ( v65 )
  {
    v20 = (__int64 *)v70;
    v21 = (__int64 *)(v74 + 1);
    if ( (unsigned __int64)(v74 + 1) > v70 )
    {
      do
      {
        v22 = *v20++;
        j_j___libc_free_0(v22, 512);
      }
      while ( v21 > v20 );
      v19 = v65;
    }
    j_j___libc_free_0(v19, 8 * v66);
  }
  return sub_21F22C0((__int64)v61);
}
