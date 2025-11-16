// Function: sub_30867A0
// Address: 0x30867a0
//
void __fastcall sub_30867A0(__int64 a1, unsigned __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // r14
  _QWORD *v5; // rax
  _QWORD *v6; // rcx
  __int64 *v7; // r14
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rdx
  __int64 *v10; // rax
  unsigned __int8 *v11; // r14
  unsigned __int8 *v12; // r14
  unsigned __int8 v13; // al
  _QWORD *v14; // r15
  unsigned __int64 v15; // rdx
  _QWORD *v16; // rax
  char v17; // bl
  __int64 v18; // rax
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  char v24; // r15
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rbx
  __int64 *v28; // r15
  unsigned __int64 v29; // rcx
  unsigned __int64 v30; // rdx
  __int64 *v31; // rax
  bool v32; // r10
  __int64 v33; // rax
  _QWORD *v34; // rax
  __int64 v35; // rax
  unsigned __int64 v36; // rcx
  __int64 *v37; // r15
  __int64 *i; // rbx
  unsigned __int64 v39; // rdx
  __int64 *v40; // rax
  char v41; // r15
  __int64 v42; // rax
  _QWORD *v43; // rax
  unsigned __int64 v44; // rbx
  unsigned __int64 v45; // rdx
  __int64 *v46; // rax
  char v47; // r14
  __int64 v48; // rax
  _QWORD *v49; // r15
  __int64 v50; // rax
  _QWORD *v51; // [rsp+8h] [rbp-D8h]
  char v52; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v53; // [rsp+10h] [rbp-D0h]
  _QWORD *v54; // [rsp+18h] [rbp-C8h]
  __int64 v55; // [rsp+18h] [rbp-C8h]
  unsigned __int64 v56; // [rsp+18h] [rbp-C8h]
  _QWORD v57[2]; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v58; // [rsp+38h] [rbp-A8h] BYREF
  __int64 *v59; // [rsp+40h] [rbp-A0h]
  __int64 *v60; // [rsp+48h] [rbp-98h]
  __int64 *v61; // [rsp+50h] [rbp-90h]
  __int64 v62; // [rsp+58h] [rbp-88h]
  __int64 v63; // [rsp+60h] [rbp-80h] BYREF
  __int64 v64; // [rsp+68h] [rbp-78h]
  _QWORD *v65; // [rsp+70h] [rbp-70h]
  _QWORD *v66; // [rsp+78h] [rbp-68h]
  _QWORD *v67; // [rsp+80h] [rbp-60h]
  unsigned __int64 v68; // [rsp+88h] [rbp-58h]
  _QWORD *v69; // [rsp+90h] [rbp-50h]
  _QWORD *v70; // [rsp+98h] [rbp-48h]
  _QWORD *v71; // [rsp+A0h] [rbp-40h]
  _QWORD *v72; // [rsp+A8h] [rbp-38h]

  v60 = &v58;
  LODWORD(v58) = 0;
  v59 = 0;
  v61 = &v58;
  v62 = 0;
  v63 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v64 = 8;
  v63 = sub_22077B0(0x40u);
  v4 = (_QWORD *)(v63 + ((4 * v64 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v5 = (_QWORD *)sub_22077B0(0x200u);
  v68 = (unsigned __int64)v4;
  v6 = v5;
  *v4 = v5;
  v66 = v5;
  v67 = v5 + 64;
  v72 = v4;
  v70 = v5;
  v71 = v5 + 64;
  v65 = v5;
  if ( v5 )
    *v5 = a2;
  v7 = v59;
  v8 = (unsigned __int64)(v5 + 1);
  v69 = v5 + 1;
  if ( v59 )
  {
    while ( 1 )
    {
      v9 = v7[4];
      v10 = (__int64 *)v7[3];
      if ( a2 < v9 )
        v10 = (__int64 *)v7[2];
      if ( !v10 )
        break;
      v7 = v10;
    }
    if ( a2 < v9 )
    {
      if ( v60 != v7 )
        goto LABEL_36;
    }
    else if ( v9 >= a2 )
    {
      goto LABEL_11;
    }
LABEL_37:
    v24 = 1;
    if ( v7 != &v58 )
      v24 = a2 < v7[4];
    goto LABEL_39;
  }
  v7 = &v58;
  if ( v60 != &v58 )
  {
LABEL_36:
    v54 = v6;
    v23 = sub_220EF80((__int64)v7);
    v6 = v54;
    if ( *(_QWORD *)(v23 + 32) >= a2 )
      goto LABEL_11;
    goto LABEL_37;
  }
  v24 = 1;
LABEL_39:
  v25 = sub_22077B0(0x28u);
  *(_QWORD *)(v25 + 32) = a2;
  sub_220F040(v24, v25, v7, &v58);
  v8 = (unsigned __int64)v69;
  v6 = v65;
  ++v62;
LABEL_11:
  v51 = a3 + 1;
  if ( (_QWORD *)v8 != v6 )
  {
    while ( 1 )
    {
      if ( v70 == (_QWORD *)v8 )
      {
        v11 = *(unsigned __int8 **)(*(v72 - 1) + 504LL);
        j_j___libc_free_0(v8);
        v26 = *--v72 + 512LL;
        v70 = (_QWORD *)*v72;
        v71 = (_QWORD *)v26;
        v69 = v70 + 63;
      }
      else
      {
        v11 = *(unsigned __int8 **)(v8 - 8);
        v69 = (_QWORD *)(v8 - 8);
      }
      v12 = sub_CEFC00(v11, (unsigned __int8 **)1);
      v13 = *v12;
      if ( *v12 > 0x1Cu )
      {
        if ( v13 == 84 )
        {
          if ( (*((_DWORD *)v12 + 1) & 0x7FFFFFF) == 0 )
            goto LABEL_27;
          v55 = 32LL * (*((_DWORD *)v12 + 1) & 0x7FFFFFF);
          v27 = 0;
          while ( 2 )
          {
            v28 = v59;
            v29 = *(_QWORD *)(*((_QWORD *)v12 - 1) + v27);
            v57[0] = v29;
            if ( !v59 )
            {
              v28 = &v58;
              if ( v60 == &v58 )
              {
                v32 = 1;
                goto LABEL_56;
              }
LABEL_63:
              v53 = v29;
              v35 = sub_220EF80((__int64)v28);
              v29 = v53;
              if ( *(_QWORD *)(v35 + 32) < v53 )
              {
                v32 = 1;
                if ( v28 != &v58 )
LABEL_65:
                  v32 = v29 < v28[4];
LABEL_56:
                v52 = v32;
                v33 = sub_22077B0(0x28u);
                *(_QWORD *)(v33 + 32) = v57[0];
                sub_220F040(v52, v33, v28, &v58);
                v34 = v69;
                ++v62;
                if ( v69 == v71 - 1 )
                {
                  sub_30865A0((unsigned __int64 *)&v63, v57);
                }
                else
                {
                  if ( v69 )
                  {
                    *v69 = v57[0];
                    v34 = v69;
                  }
                  v69 = v34 + 1;
                }
              }
LABEL_60:
              v27 += 32;
              if ( v55 == v27 )
                goto LABEL_27;
              continue;
            }
            break;
          }
          while ( 1 )
          {
            v30 = v28[4];
            v31 = (__int64 *)v28[3];
            if ( v29 < v30 )
              v31 = (__int64 *)v28[2];
            if ( !v31 )
              break;
            v28 = v31;
          }
          if ( v29 < v30 )
          {
            if ( v60 != v28 )
              goto LABEL_63;
          }
          else if ( v30 >= v29 )
          {
            goto LABEL_60;
          }
          v32 = 1;
          if ( v28 != &v58 )
            goto LABEL_65;
          goto LABEL_56;
        }
        if ( v13 == 86 )
          break;
      }
      v14 = (_QWORD *)a3[2];
      if ( !v14 )
      {
        v14 = a3 + 1;
        if ( v51 == (_QWORD *)a3[3] )
        {
          v14 = a3 + 1;
          v17 = 1;
          goto LABEL_26;
        }
LABEL_42:
        if ( (unsigned __int64)v12 <= *(_QWORD *)(sub_220EF80((__int64)v14) + 32) )
          goto LABEL_27;
        v17 = 1;
        if ( v14 == v51 )
          goto LABEL_26;
        goto LABEL_44;
      }
      while ( 1 )
      {
        v15 = v14[4];
        v16 = (_QWORD *)v14[3];
        if ( (unsigned __int64)v12 < v15 )
          v16 = (_QWORD *)v14[2];
        if ( !v16 )
          break;
        v14 = v16;
      }
      if ( (unsigned __int64)v12 < v15 )
      {
        if ( (_QWORD *)a3[3] != v14 )
          goto LABEL_42;
LABEL_25:
        v17 = 1;
        if ( v14 == v51 )
        {
LABEL_26:
          v18 = sub_22077B0(0x28u);
          *(_QWORD *)(v18 + 32) = v12;
          sub_220F040(v17, v18, v14, v51);
          ++a3[5];
          goto LABEL_27;
        }
LABEL_44:
        v17 = (unsigned __int64)v12 < v14[4];
        goto LABEL_26;
      }
      if ( v15 < (unsigned __int64)v12 )
        goto LABEL_25;
LABEL_27:
      v8 = (unsigned __int64)v69;
LABEL_28:
      if ( v65 == (_QWORD *)v8 )
        goto LABEL_29;
    }
    v36 = *((_QWORD *)v12 - 8);
    v37 = v59;
    v57[0] = v36;
    if ( v59 )
    {
      for ( i = v59; ; i = v40 )
      {
        v39 = i[4];
        v40 = (__int64 *)i[3];
        if ( v36 < v39 )
          v40 = (__int64 *)i[2];
        if ( !v40 )
          break;
      }
      if ( v36 >= v39 )
      {
        if ( v36 <= v39 )
          goto LABEL_82;
        goto LABEL_76;
      }
      if ( v60 == i )
      {
LABEL_76:
        v41 = 1;
        if ( i != &v58 )
          v41 = v36 < i[4];
        goto LABEL_78;
      }
    }
    else
    {
      i = &v58;
      if ( v60 == &v58 )
      {
        v41 = 1;
LABEL_78:
        v42 = sub_22077B0(0x28u);
        *(_QWORD *)(v42 + 32) = v57[0];
        sub_220F040(v41, v42, i, &v58);
        v43 = v69;
        ++v62;
        if ( v69 == v71 - 1 )
        {
          sub_30865A0((unsigned __int64 *)&v63, v57);
          v37 = v59;
        }
        else
        {
          if ( v69 )
          {
            *v69 = v57[0];
            v43 = v69;
          }
          v37 = v59;
          v69 = v43 + 1;
        }
LABEL_82:
        v44 = *((_QWORD *)v12 - 4);
        v57[0] = v44;
        if ( !v37 )
        {
LABEL_105:
          v37 = &v58;
          if ( v60 == &v58 )
          {
            v47 = 1;
            goto LABEL_91;
          }
LABEL_99:
          if ( *(_QWORD *)(sub_220EF80((__int64)v37) + 32) < v44 )
          {
            v47 = 1;
            if ( v37 == &v58 )
              goto LABEL_91;
            goto LABEL_107;
          }
          goto LABEL_27;
        }
        while ( 1 )
        {
LABEL_85:
          v45 = v37[4];
          v46 = (__int64 *)v37[3];
          if ( v44 < v45 )
            v46 = (__int64 *)v37[2];
          if ( !v46 )
            break;
          v37 = v46;
        }
        if ( v44 >= v45 )
        {
          if ( v45 < v44 )
            goto LABEL_90;
          goto LABEL_27;
        }
        if ( v60 == v37 )
        {
LABEL_90:
          v47 = 1;
          if ( v37 == &v58 )
            goto LABEL_91;
LABEL_107:
          v47 = v44 < v37[4];
LABEL_91:
          v48 = sub_22077B0(0x28u);
          *(_QWORD *)(v48 + 32) = v57[0];
          sub_220F040(v47, v48, v37, &v58);
          v49 = v69;
          ++v62;
          if ( v69 == v71 - 1 )
          {
            sub_30865A0((unsigned __int64 *)&v63, v57);
            v8 = (unsigned __int64)v69;
          }
          else
          {
            if ( v69 )
            {
              *v69 = v57[0];
              v49 = v69;
            }
            v8 = (unsigned __int64)(v49 + 1);
            v69 = (_QWORD *)v8;
          }
          goto LABEL_28;
        }
        goto LABEL_99;
      }
    }
    v56 = v36;
    v50 = sub_220EF80((__int64)i);
    v36 = v56;
    if ( *(_QWORD *)(v50 + 32) >= v56 )
    {
      v44 = *((_QWORD *)v12 - 4);
      v57[0] = v44;
      if ( !v37 )
        goto LABEL_105;
      goto LABEL_85;
    }
    goto LABEL_76;
  }
LABEL_29:
  v19 = v63;
  if ( v63 )
  {
    v20 = (unsigned __int64 *)v68;
    v21 = v72 + 1;
    if ( (unsigned __int64)(v72 + 1) > v68 )
    {
      do
      {
        v22 = *v20++;
        j_j___libc_free_0(v22);
      }
      while ( v21 > v20 );
      v19 = v63;
    }
    j_j___libc_free_0(v19);
  }
  sub_3085300((unsigned __int64)v59);
}
