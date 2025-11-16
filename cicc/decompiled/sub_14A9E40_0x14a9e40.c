// Function: sub_14A9E40
// Address: 0x14a9e40
//
__int64 __fastcall sub_14A9E40(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rdx
  _BYTE *v12; // rax
  __int64 v13; // rcx
  char v14; // al
  char v15; // al
  __int64 v16; // rcx
  char v17; // al
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // r8
  char v23; // al
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rdx
  char v27; // al
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 v30; // rdx
  unsigned int v31; // r12d
  __int64 v32; // r13
  __int64 v33; // rbx
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r14
  __int64 v39; // rbx
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // r15
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // [rsp+0h] [rbp-90h]
  __int64 v46; // [rsp+8h] [rbp-88h]
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+10h] [rbp-80h]
  __int64 v49; // [rsp+10h] [rbp-80h]
  __int64 v50; // [rsp+10h] [rbp-80h]
  __int64 v51; // [rsp+10h] [rbp-80h]
  __int64 v52; // [rsp+18h] [rbp-78h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  __int64 v55; // [rsp+18h] [rbp-78h]
  __int64 v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  __int64 v58; // [rsp+18h] [rbp-78h]
  __int64 v59; // [rsp+18h] [rbp-78h]
  __int64 v60; // [rsp+18h] [rbp-78h]
  __int64 v61; // [rsp+18h] [rbp-78h]
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+18h] [rbp-78h]
  __int64 v64; // [rsp+18h] [rbp-78h]
  __int64 v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+18h] [rbp-78h]
  __int64 v67; // [rsp+18h] [rbp-78h]
  char v68[8]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v69; // [rsp+28h] [rbp-68h] BYREF
  __int64 v70; // [rsp+30h] [rbp-60h]
  char v71[8]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v72; // [rsp+48h] [rbp-48h] BYREF
  __int64 v73; // [rsp+50h] [rbp-40h]

  v3 = a1 + 8;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_1698270(a1, a2);
  v8 = sub_16982C0(a1, a2, v6, v7);
  v9 = a2 + 8;
  v10 = v8;
  if ( v5 != v4 )
    goto LABEL_2;
  v48 = v8;
  v12 = (_BYTE *)sub_16D40F0(qword_4FBB490);
  v9 = a2 + 8;
  v10 = v48;
  v14 = v12 ? *v12 : LOBYTE(qword_4FBB490[2]);
  v4 = *(_QWORD *)(a1 + 8);
  if ( !v14 )
    goto LABEL_2;
  if ( v4 == v48 )
  {
    v15 = sub_16A0F40(v3, a2, v48, v13, v9);
    v10 = v48;
    v9 = a2 + 8;
  }
  else
  {
    v15 = sub_16984B0(v3, a2, v48, v13, v9);
    v9 = a2 + 8;
    v10 = v48;
  }
  if ( !v15 )
  {
    v49 = v10;
    v52 = v9;
    if ( *(_QWORD *)(a2 + 8) == v10 )
    {
      v17 = sub_16A0F40(v9, a2, v10, v16, v9);
      v10 = v49;
      v9 = v52;
    }
    else
    {
      v17 = sub_16984B0(v9, a2, v10, v16, v9);
      v9 = v52;
      v10 = v49;
    }
    if ( !v17 )
    {
      v4 = *(_QWORD *)(a1 + 8);
LABEL_2:
      if ( v4 == v10 )
        return sub_16A1380(v3, v9);
      else
        return sub_1699540(v3, v9);
    }
  }
  v50 = v10;
  v53 = v9;
  if ( *(_QWORD *)(a1 + 8) == v10 )
  {
    sub_169C6E0(&v69, v3);
    v19 = v50;
    v18 = v53;
  }
  else
  {
    sub_16986C0(&v69, v3);
    v18 = v53;
    v19 = v50;
  }
  v54 = v19;
  v20 = v18;
  if ( *(_QWORD *)(a2 + 8) == v19 )
    sub_169C6E0(&v72, v18);
  else
    sub_16986C0(&v72, v18);
  if ( v69 == v54 )
    v23 = sub_16A0F40(&v69, v20, v54, v21, v22);
  else
    v23 = sub_16984B0(&v69, v20, v54, v21, v22);
  v26 = v54;
  if ( v23 )
  {
    v20 = 0;
    if ( v69 == v54 )
      sub_169C980(&v69, 0);
    else
      sub_169B620(&v69, 0);
    v26 = v54;
  }
  v55 = v26;
  if ( v72 == v26 )
    v27 = sub_16A0F40(&v72, v20, v26, v24, v25);
  else
    v27 = sub_16984B0(&v72, v20, v26, v24, v25);
  v28 = v55;
  if ( v27 )
  {
    if ( v72 == v55 )
      sub_169C980(&v72, 0);
    else
      sub_169B620(&v72, 0);
    v28 = v55;
  }
  v56 = v28;
  v29 = sub_14A9E40(v68, v71);
  v30 = v56;
  v31 = v29;
  if ( v72 == v56 )
  {
    v38 = v73;
    if ( v73 )
    {
      v39 = v73 + 32LL * *(_QWORD *)(v73 - 8);
      if ( v73 != v39 )
      {
        do
        {
          v39 -= 32;
          if ( *(_QWORD *)(v39 + 8) == v30 )
          {
            v40 = *(_QWORD *)(v39 + 16);
            v45 = v40;
            if ( v40 )
            {
              v41 = 32LL * *(_QWORD *)(v40 - 8);
              v42 = v40 + v41;
              if ( v40 != v40 + v41 )
              {
                do
                {
                  v42 -= 32;
                  if ( *(_QWORD *)(v42 + 8) == v30 )
                  {
                    v43 = *(_QWORD *)(v42 + 16);
                    if ( v43 )
                    {
                      v44 = v43 + 32LL * *(_QWORD *)(v43 - 8);
                      if ( v43 != v44 )
                      {
                        do
                        {
                          v47 = v30;
                          v63 = v44 - 32;
                          sub_127D120((_QWORD *)(v44 - 24));
                          v44 = v63;
                          v30 = v47;
                        }
                        while ( v43 != v63 );
                      }
                      v64 = v30;
                      j_j_j___libc_free_0_0(v43 - 8);
                      v30 = v64;
                    }
                  }
                  else
                  {
                    v62 = v30;
                    sub_1698460(v42 + 8);
                    v30 = v62;
                  }
                }
                while ( v45 != v42 );
              }
              v67 = v30;
              j_j_j___libc_free_0_0(v45 - 8);
              v30 = v67;
            }
          }
          else
          {
            v61 = v30;
            sub_1698460(v39 + 8);
            v30 = v61;
          }
        }
        while ( v38 != v39 );
      }
      v65 = v30;
      j_j_j___libc_free_0_0(v38 - 8);
      v30 = v65;
    }
  }
  else
  {
    sub_1698460(&v72);
    v30 = v56;
  }
  if ( v69 == v30 )
  {
    v32 = v70;
    if ( v70 )
    {
      v33 = v70 + 32LL * *(_QWORD *)(v70 - 8);
      if ( v70 != v33 )
      {
        do
        {
          v33 -= 32;
          if ( *(_QWORD *)(v33 + 8) == v30 )
          {
            v34 = *(_QWORD *)(v33 + 16);
            if ( v34 )
            {
              v35 = v34 + 32LL * *(_QWORD *)(v34 - 8);
              while ( v34 != v35 )
              {
                v35 -= 32;
                if ( *(_QWORD *)(v35 + 8) == v30 )
                {
                  v36 = *(_QWORD *)(v35 + 16);
                  if ( v36 )
                  {
                    v37 = v36 + 32LL * *(_QWORD *)(v36 - 8);
                    if ( v36 != v37 )
                    {
                      do
                      {
                        v46 = v30;
                        v51 = v36;
                        v59 = v37 - 32;
                        sub_127D120((_QWORD *)(v37 - 24));
                        v37 = v59;
                        v36 = v51;
                        v30 = v46;
                      }
                      while ( v51 != v59 );
                    }
                    v60 = v30;
                    j_j_j___libc_free_0_0(v36 - 8);
                    v30 = v60;
                  }
                }
                else
                {
                  v58 = v30;
                  sub_1698460(v35 + 8);
                  v30 = v58;
                }
              }
              v66 = v30;
              j_j_j___libc_free_0_0(v34 - 8);
              v30 = v66;
            }
          }
          else
          {
            v57 = v30;
            sub_1698460(v33 + 8);
            v30 = v57;
          }
        }
        while ( v32 != v33 );
      }
      j_j_j___libc_free_0_0(v32 - 8);
    }
  }
  else
  {
    sub_1698460(&v69);
  }
  return v31;
}
