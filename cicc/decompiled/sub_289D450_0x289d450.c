// Function: sub_289D450
// Address: 0x289d450
//
void __fastcall sub_289D450(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r12
  char v3; // al
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rbx
  _QWORD *v7; // r15
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rsi
  _BYTE *v11; // rdx
  _BYTE *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r9
  __int64 v15; // rcx
  _QWORD *v16; // rdx
  unsigned __int8 *v17; // rax
  _BYTE *v18; // r9
  __int64 v19; // r9
  bool v20; // cc
  __int64 v21; // rcx
  _QWORD *v22; // rax
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // r9
  __int64 v29; // rcx
  _QWORD *v30; // rdx
  unsigned __int8 *v31; // rax
  unsigned int v32; // [rsp+10h] [rbp-1B0h]
  const char *v33; // [rsp+18h] [rbp-1A8h]
  _BYTE *v34; // [rsp+20h] [rbp-1A0h]
  unsigned int v35; // [rsp+28h] [rbp-198h]
  __int64 v36; // [rsp+30h] [rbp-190h]
  __int64 v37; // [rsp+38h] [rbp-188h]
  __int64 v38; // [rsp+40h] [rbp-180h]
  __int64 v39; // [rsp+48h] [rbp-178h]
  _BYTE *v40; // [rsp+48h] [rbp-178h]
  unsigned int v41; // [rsp+48h] [rbp-178h]
  __int64 v42; // [rsp+48h] [rbp-178h]
  _BYTE *v43; // [rsp+48h] [rbp-178h]
  _BYTE *v44; // [rsp+48h] [rbp-178h]
  _BYTE *v45; // [rsp+58h] [rbp-168h] BYREF
  char *v46; // [rsp+60h] [rbp-160h] BYREF
  __int64 v47; // [rsp+68h] [rbp-158h] BYREF
  __int64 v48; // [rsp+70h] [rbp-150h] BYREF
  unsigned int **v49; // [rsp+78h] [rbp-148h] BYREF
  _QWORD v50[4]; // [rsp+80h] [rbp-140h] BYREF
  unsigned int **v51[4]; // [rsp+A0h] [rbp-120h] BYREF
  __int16 v52; // [rsp+C0h] [rbp-100h]
  const char *v53; // [rsp+D0h] [rbp-F0h] BYREF
  _BYTE *v54; // [rsp+D8h] [rbp-E8h]
  _BYTE **v55; // [rsp+E0h] [rbp-E0h]
  __int64 v56; // [rsp+E8h] [rbp-D8h]
  __int64 v57; // [rsp+F0h] [rbp-D0h]
  unsigned int *v58; // [rsp+100h] [rbp-C0h] BYREF
  int v59; // [rsp+108h] [rbp-B8h]
  char **v60; // [rsp+110h] [rbp-B0h]
  unsigned int v61; // [rsp+118h] [rbp-A8h]
  __int64 *v62; // [rsp+120h] [rbp-A0h]
  unsigned int v63; // [rsp+128h] [rbp-98h]
  __int64 *v64; // [rsp+130h] [rbp-90h]

  v2 = (_QWORD *)a2;
  v3 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 == 85 )
  {
    v4 = *(_QWORD *)(a2 - 32);
    if ( !v4 )
      return;
    if ( *(_BYTE *)v4 )
      return;
    if ( *(_QWORD *)(v4 + 24) != *(_QWORD *)(a2 + 80) )
      return;
    if ( *(_DWORD *)(v4 + 36) != 233 )
      return;
    v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    v6 = *(_QWORD **)(a2 - 32 * v5);
    if ( !v6 )
      return;
    v7 = *(_QWORD **)(a2 + 32 * (1 - v5));
    if ( !v7 )
      return;
    if ( **(_BYTE **)(a2 + 32 * (2 - v5)) != 17 )
      return;
    v47 = *(_QWORD *)(a2 + 32 * (2 - v5));
    v39 = *(_QWORD *)(a2 + 32 * (3 - v5));
    if ( *(_BYTE *)v39 != 17 || **(_BYTE **)(a2 + 32 * (4 - v5)) != 17 )
      return;
    v48 = *(_QWORD *)(a2 + 32 * (4 - v5));
    v55 = &v45;
    LODWORD(v53) = 234;
    LODWORD(v54) = 0;
    if ( (unsigned __int8)sub_10E25C0((__int64)&v53, (__int64)v6) )
    {
      LODWORD(v58) = 234;
      v59 = 0;
      v60 = &v46;
      if ( (unsigned __int8)sub_10E25C0((__int64)&v58, (__int64)v7) )
      {
        sub_23D0AB0((__int64)&v58, a2, 0, 0, 0);
        v52 = 257;
        v20 = *(_DWORD *)(v47 + 32) <= 0x40u;
        v49 = &v58;
        if ( v20 )
          v21 = *(_QWORD *)(v47 + 24);
        else
          v21 = **(_QWORD **)(v47 + 24);
        v22 = *(_QWORD **)(v39 + 24);
        if ( *(_DWORD *)(v39 + 32) > 0x40u )
          v22 = (_QWORD *)*v22;
        v41 = (unsigned int)v22;
        v23 = *(_QWORD *)(v48 + 24);
        if ( *(_DWORD *)(v48 + 32) > 0x40u )
          v23 = **(_QWORD **)(v48 + 24);
        v32 = v21;
        v35 = v23;
        v33 = v46;
        v36 = *((_QWORD *)v46 + 1);
        v34 = v45;
        v38 = *((_QWORD *)v45 + 1);
        v37 = sub_BCDA70(*(__int64 **)(v36 + 24), (int)v23 * (int)v21);
        v53 = v33;
        v54 = v34;
        v24 = sub_BCB2D0(v49[9]);
        v55 = (_BYTE **)sub_ACD640(v24, v35, 0);
        v25 = sub_BCB2D0(v49[9]);
        v56 = sub_ACD640(v25, v41, 0);
        v26 = sub_BCB2D0(v49[9]);
        v57 = sub_ACD640(v26, v32, 0);
        v50[0] = v37;
        v50[2] = v38;
        v50[1] = v36;
        v27 = sub_B6E160(*(__int64 **)(*((_QWORD *)v49[6] + 9) + 40LL), 0xE9u, (__int64)v50, 3);
        sub_921880(v49, *(_QWORD *)(v27 + 24), v27, (int)&v53, 5, (__int64)v51, 0);
        sub_28940A0((__int64)&v53, v48, v47);
        v42 = v28;
        sub_2896BA0(a1, v28, (__int64)v53, (int)v54);
        LOWORD(v57) = 257;
        if ( *(_DWORD *)(v47 + 32) <= 0x40u )
          v29 = *(_QWORD *)(v47 + 24);
        else
          v29 = **(_QWORD **)(v47 + 24);
        v30 = *(_QWORD **)(v48 + 24);
        if ( *(_DWORD *)(v48 + 32) > 0x40u )
          v30 = (_QWORD *)*v30;
        v31 = (unsigned __int8 *)sub_289B7C0(&v49, v42, (unsigned int)v30, v29, (__int64)&v53);
        sub_2896320(a1, a2, v31);
        if ( !*(_QWORD *)(a2 + 16) )
          sub_28957D0(a1, (_QWORD *)a2);
        if ( !v6[2] )
        {
          a2 = (__int64)v6;
          sub_28957D0(a1, v6);
        }
        if ( v6 != v7 && !v7[2] )
        {
          a2 = (__int64)v7;
          sub_28957D0(a1, v7);
        }
LABEL_39:
        sub_F94A20(&v58, a2);
        return;
      }
    }
    v3 = *(_BYTE *)a2;
  }
  if ( v3 == 43 )
  {
    v8 = *(_QWORD *)(a2 - 64);
    if ( v8 )
    {
      v9 = *(_QWORD *)(a2 - 32);
      if ( v9 )
      {
        v10 = *(_QWORD *)(a2 - 64);
        LODWORD(v58) = 234;
        v60 = &v45;
        v62 = &v47;
        v59 = 0;
        v61 = 1;
        v63 = 2;
        v64 = &v48;
        if ( (unsigned __int8)sub_10E25C0((__int64)&v58, v10) )
        {
          if ( *(_BYTE *)v8 == 85 )
          {
            v11 = *(_BYTE **)(v8 + 32 * (v61 - (unsigned __int64)(*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
            if ( *v11 == 17 )
            {
              *v62 = (__int64)v11;
              if ( *(_BYTE *)v8 == 85 )
              {
                v12 = *(_BYTE **)(v8 + 32 * (v63 - (unsigned __int64)(*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
                if ( *v12 == 17 )
                {
                  *v64 = (__int64)v12;
                  LODWORD(v53) = 234;
                  LODWORD(v54) = 0;
                  v55 = &v46;
                  LODWORD(v56) = 1;
                  LODWORD(v57) = 2;
                  if ( (unsigned __int8)sub_10E25C0((__int64)&v53, v9) )
                  {
                    if ( *(_BYTE *)v9 == 85 )
                    {
                      v13 = *(_DWORD *)(v9 + 4) & 0x7FFFFFF;
                      if ( **(_BYTE **)(v9 + 32 * ((unsigned int)v56 - v13)) == 17
                        && **(_BYTE **)(v9 + 32 * ((unsigned int)v57 - v13)) == 17 )
                      {
                        sub_23D0AB0((__int64)&v58, (__int64)v2, 0, 0, 0);
                        HIDWORD(v51[0]) = 0;
                        v53 = "mfadd";
                        LOWORD(v57) = 259;
                        v14 = sub_92A220(&v58, v45, v46, LODWORD(v51[0]), (__int64)&v53, 0);
                        LOWORD(v57) = 259;
                        v53 = "mfadd_t";
                        v51[0] = &v58;
                        if ( *(_DWORD *)(v48 + 32) <= 0x40u )
                          v15 = *(_QWORD *)(v48 + 24);
                        else
                          v15 = **(_QWORD **)(v48 + 24);
                        v16 = *(_QWORD **)(v47 + 24);
                        if ( *(_DWORD *)(v47 + 32) > 0x40u )
                          v16 = (_QWORD *)*v16;
                        v40 = (_BYTE *)v14;
                        v17 = (unsigned __int8 *)sub_289B7C0(v51, v14, (unsigned int)v16, v15, (__int64)&v53);
                        a2 = (__int64)v2;
                        sub_2896320(a1, (__int64)v2, v17);
                        v18 = v40;
                        if ( !v2[2] )
                        {
                          a2 = (__int64)v2;
                          sub_28957D0(a1, v2);
                          v18 = v40;
                        }
                        if ( !*(_QWORD *)(v8 + 16) )
                        {
                          a2 = v8;
                          v43 = v18;
                          sub_28957D0(a1, (_QWORD *)v8);
                          v18 = v43;
                        }
                        if ( v8 != v9 && !*(_QWORD *)(v9 + 16) )
                        {
                          a2 = v9;
                          v44 = v18;
                          sub_28957D0(a1, (_QWORD *)v9);
                          v18 = v44;
                        }
                        if ( *v18 > 0x1Cu )
                        {
                          sub_28940A0((__int64)&v53, v47, v48);
                          a2 = v19;
                          sub_2896BA0(a1, v19, (__int64)v53, (int)v54);
                        }
                        goto LABEL_39;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
