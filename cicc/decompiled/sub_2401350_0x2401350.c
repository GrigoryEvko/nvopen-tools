// Function: sub_2401350
// Address: 0x2401350
//
__int64 __fastcall sub_2401350(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 **v6; // r15
  unsigned __int8 **v7; // rbx
  __int64 *v8; // r12
  unsigned __int8 *v11; // rdi
  char v12; // al
  __int64 v13; // rsi
  __int64 v14; // rdx
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // rbx
  unsigned __int64 *v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // r14
  bool v21; // r9
  __int64 v22; // rax
  _QWORD *v23; // rax
  __int64 *v24; // rdx
  __int64 *v25; // rax
  __int64 v26; // r14
  bool v27; // r9
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 *v30; // rdx
  unsigned __int64 *v31; // rax
  __int64 *v32; // r10
  __int64 *v33; // r13
  unsigned __int64 *v34; // rbx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rbx
  _QWORD *v40; // r13
  __int64 v41; // rax
  _QWORD *v42; // rbx
  _QWORD *v43; // r12
  char v45; // [rsp+18h] [rbp-F8h]
  __int64 *v46; // [rsp+18h] [rbp-F8h]
  __int64 *v47; // [rsp+20h] [rbp-F0h]
  char v48; // [rsp+20h] [rbp-F0h]
  __int64 *v49; // [rsp+28h] [rbp-E8h]
  __int64 *v50; // [rsp+28h] [rbp-E8h]
  __int64 *v51; // [rsp+28h] [rbp-E8h]
  __int64 *v52; // [rsp+28h] [rbp-E8h]
  __int64 *v53; // [rsp+28h] [rbp-E8h]
  unsigned __int64 *v55; // [rsp+30h] [rbp-E0h]
  unsigned __int64 *v56; // [rsp+30h] [rbp-E0h]
  unsigned __int64 *v57; // [rsp+30h] [rbp-E0h]
  unsigned __int64 *v59; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 *v60; // [rsp+48h] [rbp-C8h]
  unsigned __int64 *v61; // [rsp+50h] [rbp-C0h]
  __int64 v62; // [rsp+60h] [rbp-B0h] BYREF
  _QWORD *v63; // [rsp+68h] [rbp-A8h]
  __int64 v64; // [rsp+70h] [rbp-A0h]
  unsigned int v65; // [rsp+78h] [rbp-98h]
  __int64 v66; // [rsp+80h] [rbp-90h] BYREF
  __int64 v67; // [rsp+88h] [rbp-88h] BYREF
  unsigned __int64 v68; // [rsp+90h] [rbp-80h]
  __int64 *v69; // [rsp+98h] [rbp-78h]
  __int64 *v70; // [rsp+A0h] [rbp-70h]
  __int64 v71; // [rsp+A8h] [rbp-68h]
  __int64 v72; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v73; // [rsp+B8h] [rbp-58h] BYREF
  unsigned __int64 v74; // [rsp+C0h] [rbp-50h]
  __int64 *v75; // [rsp+C8h] [rbp-48h]
  __int64 *v76; // [rsp+D0h] [rbp-40h]
  __int64 v77; // [rsp+D8h] [rbp-38h]

  v6 = *(unsigned __int8 ***)(a3 + 8);
  v7 = &v6[*(unsigned int *)(a3 + 24)];
  if ( *(_DWORD *)(a3 + 16) && v6 != v7 )
  {
    while ( *v6 == (unsigned __int8 *)-8192LL || *v6 == (unsigned __int8 *)-4096LL )
    {
      if ( ++v6 == v7 )
        goto LABEL_2;
    }
    if ( v6 != v7 )
    {
LABEL_11:
      v11 = *v6;
      v72 = 0;
      v73 = 0;
      v74 = 0;
      LODWORD(v75) = 0;
      v12 = sub_24005F0(v11, a1, a4, a5, 0, (__int64)&v72);
      v13 = 16LL * (unsigned int)v75;
      if ( !v12 )
      {
        sub_C7D6A0(v73, v13, 8);
        LODWORD(v8) = 1;
        return (unsigned int)v8;
      }
      sub_C7D6A0(v73, v13, 8);
      while ( ++v6 != v7 )
      {
        if ( *v6 != (unsigned __int8 *)-8192LL && *v6 != (unsigned __int8 *)-4096LL )
        {
          if ( v6 != v7 )
            goto LABEL_11;
          break;
        }
      }
    }
  }
LABEL_2:
  LODWORD(v8) = 0;
  if ( *(_DWORD *)(a2 + 16) && *(_DWORD *)(a3 + 16) )
  {
    LODWORD(v67) = 0;
    v75 = &v73;
    v14 = *(unsigned int *)(a2 + 24);
    v76 = &v73;
    v15 = *(unsigned __int64 **)(a2 + 8);
    v68 = 0;
    v69 = &v67;
    v70 = &v67;
    v71 = 0;
    LODWORD(v73) = 0;
    v74 = 0;
    v77 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v55 = &v15[v14];
    if ( v15 == v55 )
      goto LABEL_23;
    while ( 1 )
    {
      v16 = *v15;
      v17 = v15;
      if ( *v15 != -8192 && v16 != -4096 )
        break;
      if ( v55 == ++v15 )
        goto LABEL_23;
    }
    if ( v55 == v15 )
    {
LABEL_23:
      v18 = *(unsigned __int64 **)(a3 + 8);
      v56 = &v18[*(unsigned int *)(a3 + 24)];
    }
    else
    {
      v8 = &v66;
      do
      {
        v25 = sub_23FFFB0(v16, a4, (__int64)&v62);
        v26 = v25[3];
        v50 = v25 + 1;
        if ( (__int64 *)v26 != v25 + 1 )
        {
          do
          {
            v29 = sub_23FE670(&v66, (__int64)&v67, (unsigned __int64 *)(v26 + 32));
            if ( v30 )
            {
              v27 = v29 || v30 == &v67 || *(_QWORD *)(v26 + 32) < (unsigned __int64)v30[4];
              v46 = v30;
              v48 = v27;
              v28 = sub_22077B0(0x28u);
              *(_QWORD *)(v28 + 32) = *(_QWORD *)(v26 + 32);
              sub_220F040(v48, v28, v46, &v67);
              ++v71;
            }
            v26 = sub_220EF30(v26);
          }
          while ( v50 != (__int64 *)v26 );
        }
        v31 = v17 + 1;
        if ( v17 + 1 == v55 )
          break;
        while ( 1 )
        {
          v16 = *v31;
          v17 = v31;
          if ( *v31 != -8192 && v16 != -4096 )
            break;
          if ( v55 == ++v31 )
            goto LABEL_55;
        }
      }
      while ( v31 != v55 );
LABEL_55:
      v18 = *(unsigned __int64 **)(a3 + 8);
      v56 = &v18[*(unsigned int *)(a3 + 24)];
      if ( !*(_DWORD *)(a3 + 16) )
      {
LABEL_56:
        v59 = 0;
        v32 = v75;
        v60 = 0;
        v33 = v69;
        v61 = 0;
        if ( v75 != &v73 )
        {
          v57 = 0;
          v34 = 0;
          if ( v69 != &v67 )
          {
            do
            {
              v36 = v33[4];
              if ( v36 < v32[4] )
              {
                v51 = v32;
                v35 = sub_220EF30((__int64)v33);
                v32 = v51;
                v33 = (__int64 *)v35;
              }
              else
              {
                if ( v36 <= v32[4] )
                {
                  if ( v57 == v34 )
                  {
                    v53 = v32;
                    sub_9281F0((__int64)&v59, v57, v33 + 4);
                    v34 = v60;
                    v32 = v53;
                    v57 = v61;
                  }
                  else
                  {
                    if ( v34 )
                    {
                      *v34 = v36;
                      v34 = v60;
                      v57 = v61;
                    }
                    v60 = ++v34;
                  }
                  v52 = v32;
                  v37 = sub_220EF30((__int64)v33);
                  v32 = v52;
                  v33 = (__int64 *)v37;
                }
                v32 = (__int64 *)sub_220EF30((__int64)v32);
              }
              LOBYTE(v8) = v33 != &v67 && v32 != &v73;
            }
            while ( (_BYTE)v8 );
            if ( v34 != v59 )
            {
              if ( v59 )
                j_j___libc_free_0((unsigned __int64)v59);
              v38 = v65;
              if ( v65 )
              {
                v39 = v63;
                v40 = &v63[7 * v65];
                do
                {
                  if ( *v39 != -4096 && *v39 != -8192 )
                    sub_23FBEA0(v39[3]);
                  v39 += 7;
                }
                while ( v40 != v39 );
                v38 = v65;
              }
              sub_C7D6A0((__int64)v63, 56 * v38, 8);
              sub_23FBEA0(v74);
              sub_23FBEA0(v68);
              return (unsigned int)v8;
            }
            if ( v34 )
              j_j___libc_free_0((unsigned __int64)v34);
          }
        }
        v41 = v65;
        if ( v65 )
        {
          v42 = v63;
          v43 = &v63[7 * v65];
          do
          {
            if ( *v42 != -4096 && *v42 != -8192 )
              sub_23FBEA0(v42[3]);
            v42 += 7;
          }
          while ( v43 != v42 );
          v41 = v65;
        }
        LODWORD(v8) = 1;
        sub_C7D6A0((__int64)v63, 56 * v41, 8);
        sub_23FBEA0(v74);
        sub_23FBEA0(v68);
        return (unsigned int)v8;
      }
    }
    if ( v56 != v18 )
    {
      while ( *v18 == -8192 || *v18 == -4096 )
      {
        if ( ++v18 == v56 )
          goto LABEL_56;
      }
      if ( v18 != v56 )
      {
        v8 = &v72;
LABEL_31:
        v19 = sub_23FFFB0(*v18, a4, (__int64)&v62);
        v20 = v19[3];
        v49 = v19 + 1;
        if ( (__int64 *)v20 != v19 + 1 )
        {
          do
          {
            v23 = sub_23FE670(&v72, (__int64)&v73, (unsigned __int64 *)(v20 + 32));
            if ( v24 )
            {
              v21 = v23 || v24 == &v73 || *(_QWORD *)(v20 + 32) < (unsigned __int64)v24[4];
              v45 = v21;
              v47 = v24;
              v22 = sub_22077B0(0x28u);
              *(_QWORD *)(v22 + 32) = *(_QWORD *)(v20 + 32);
              sub_220F040(v45, v22, v47, &v73);
              ++v77;
            }
            v20 = sub_220EF30(v20);
          }
          while ( v49 != (__int64 *)v20 );
        }
        while ( ++v18 != v56 )
        {
          if ( *v18 != -8192 && *v18 != -4096 )
          {
            if ( v18 != v56 )
              goto LABEL_31;
            goto LABEL_56;
          }
        }
      }
    }
    goto LABEL_56;
  }
  return (unsigned int)v8;
}
