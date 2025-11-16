// Function: sub_BC8F20
// Address: 0xbc8f20
//
void __fastcall sub_BC8F20(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  __int64 *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  _WORD *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // r14
  unsigned __int8 v13; // al
  __int64 *v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  _WORD *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned int v23; // r13d
  __int64 v24; // rsi
  __int64 v25; // r15
  __int64 v26; // rdx
  _QWORD *v27; // rsi
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  __int64 v36; // rcx
  __int64 v37; // r15
  __int64 v38; // rcx
  unsigned __int8 v39; // al
  int v40; // eax
  unsigned int v41; // eax
  unsigned __int8 v42; // dl
  __int64 v43; // rcx
  __int64 v44; // rax
  __int64 v45; // rax
  _QWORD *v46; // rsi
  unsigned int v47; // ebx
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rbx
  __int64 v54; // rax
  unsigned __int64 v55; // rdx
  unsigned int v57; // [rsp+30h] [rbp-C0h]
  __int64 v58; // [rsp+30h] [rbp-C0h]
  __int64 v59; // [rsp+38h] [rbp-B8h]
  __int64 v60; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v61; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v62; // [rsp+58h] [rbp-98h]
  __int64 v63; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v64; // [rsp+68h] [rbp-88h]
  __int64 v65; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v66; // [rsp+78h] [rbp-78h]
  __int64 *v67; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v68; // [rsp+88h] [rbp-68h]
  __int64 *v69; // [rsp+90h] [rbp-60h] BYREF
  __int64 v70; // [rsp+98h] [rbp-58h]
  _QWORD v71[10]; // [rsp+A0h] [rbp-50h] BYREF

  if ( (a1[7] & 0x20) != 0 )
  {
    v3 = sub_B91C10((__int64)a1, 2);
    v4 = v3;
    if ( v3 )
    {
      v59 = v3 - 16;
      v5 = *(_BYTE *)(v3 - 16);
      v6 = (v5 & 2) != 0 ? *(__int64 **)(v4 - 32) : (__int64 *)(v59 - 8LL * ((v5 >> 2) & 0xF));
      v7 = *v6;
      if ( !*(_BYTE *)*v6 )
      {
        if ( (v8 = sub_B91420(*v6), v9 == 14)
          && *(_QWORD *)v8 == 0x775F68636E617262LL
          && *(_DWORD *)(v8 + 8) == 1751607653
          && *(_WORD *)(v8 + 12) == 29556
          || (v10 = (_WORD *)sub_B91420(v7), v11 == 2) && *v10 == 20566 )
        {
          if ( (unsigned __int8)sub_BC8730(a1) )
          {
            v60 = sub_BD5C60(a1, 2);
            v12 = (__int64 *)v60;
            v69 = v71;
            v70 = 0x300000000LL;
            v13 = *(_BYTE *)(v4 - 16);
            if ( (v13 & 2) != 0 )
              v14 = *(__int64 **)(v4 - 32);
            else
              v14 = (__int64 *)(v59 - 8LL * ((v13 >> 2) & 0xF));
            v15 = *v14;
            LODWORD(v70) = 1;
            v62 = 128;
            v71[0] = v15;
            sub_C43690(&v61, a2, 0);
            v64 = 128;
            sub_C43690(&v63, a3, 0);
            v16 = sub_B91420(v7);
            if ( v17 == 14
              && *(_QWORD *)v16 == 0x775F68636E617262LL
              && *(_DWORD *)(v16 + 8) == 1751607653
              && *(_WORD *)(v16 + 12) == 29556
              && ((*(_BYTE *)(v4 - 16) & 2) == 0
                ? (v40 = (*(_WORD *)(v4 - 16) >> 6) & 0xF)
                : (v40 = *(_DWORD *)(v4 - 24)),
                  v40) )
            {
              v41 = sub_BC8810(v4);
              v42 = *(_BYTE *)(v4 - 16);
              if ( (v42 & 2) != 0 )
                v43 = *(_QWORD *)(v4 - 32);
              else
                v43 = v59 - 8LL * ((v42 >> 2) & 0xF);
              v44 = *(_QWORD *)(v43 + 8LL * v41);
              if ( *(_BYTE *)v44 != 1 || (v45 = *(_QWORD *)(v44 + 136), *(_BYTE *)v45 != 17) )
                BUG();
              v46 = *(_QWORD **)(v45 + 24);
              if ( *(_DWORD *)(v45 + 32) > 0x40u )
                v46 = (_QWORD *)*v46;
              v66 = 128;
              sub_C43690(&v65, v46, 0);
              sub_C47360(&v65, &v61);
              sub_C4A1D0(&v67, &v65, &v63);
              v47 = v68;
              if ( v68 > 0x40 )
              {
                v48 = 0xFFFFFFFFLL;
                if ( v47 - (unsigned int)sub_C444A0(&v67) <= 0x40 && (unsigned __int64)*v67 <= 0xFFFFFFFF )
                  v48 = *v67;
              }
              else
              {
                v48 = 0xFFFFFFFFLL;
                if ( (unsigned __int64)v67 <= 0xFFFFFFFF )
                  v48 = (__int64)v67;
              }
              v49 = sub_BCB2D0(v60);
              v50 = sub_ACD640(v49, v48, 0);
              v53 = sub_B8C140((__int64)&v60, v50, v51, v52);
              v54 = (unsigned int)v70;
              v55 = (unsigned int)v70 + 1LL;
              if ( v55 > HIDWORD(v70) )
              {
                sub_C8D5F0(&v69, v71, v55, 8);
                v54 = (unsigned int)v70;
              }
              v69[v54] = v53;
              LODWORD(v70) = v70 + 1;
              if ( v68 > 0x40 && v67 )
                j_j___libc_free_0_0(v67);
              if ( v66 > 0x40 && v65 )
                j_j___libc_free_0_0(v65);
              v21 = (unsigned int)v70;
            }
            else
            {
              v18 = (_WORD *)sub_B91420(v7);
              v20 = v19;
              v21 = (unsigned int)v70;
              if ( v20 == 2 )
              {
                v23 = 1;
                if ( *v18 == 20566 )
                {
                  while ( 1 )
                  {
                    v35 = *(_BYTE *)(v4 - 16);
                    if ( (v35 & 2) != 0 )
                    {
                      if ( v23 >= *(_DWORD *)(v4 - 24) )
                        break;
                      v36 = *(_QWORD *)(v4 - 32);
                    }
                    else
                    {
                      if ( v23 >= ((*(_WORD *)(v4 - 16) >> 6) & 0xFu) )
                        break;
                      v36 = v59 - 8LL * ((v35 >> 2) & 0xF);
                    }
                    v37 = *(_QWORD *)(v36 + 8LL * v23);
                    if ( v21 + 1 > HIDWORD(v70) )
                    {
                      sub_C8D5F0(&v69, v71, v21 + 1, 8);
                      v21 = (unsigned int)v70;
                    }
                    v69[v21] = v37;
                    v38 = (unsigned int)(v70 + 1);
                    LODWORD(v70) = v70 + 1;
                    v39 = *(_BYTE *)(v4 - 16);
                    if ( (v39 & 2) != 0 )
                      v24 = *(_QWORD *)(v4 - 32);
                    else
                      v24 = v59 - 8LL * ((v39 >> 2) & 0xF);
                    v25 = *(_QWORD *)(v24 + 8LL * (v23 + 1));
                    if ( *(_BYTE *)v25 != 1 || (v26 = *(_QWORD *)(v25 + 136), *(_BYTE *)v26 != 17) )
                      BUG();
                    v27 = *(_QWORD **)(v26 + 24);
                    if ( *(_DWORD *)(v26 + 32) > 0x40u )
                      v27 = (_QWORD *)*v27;
                    if ( v27 == (_QWORD *)-1LL )
                    {
                      if ( v38 + 1 > (unsigned __int64)HIDWORD(v70) )
                      {
                        sub_C8D5F0(&v69, v71, v38 + 1, 8);
                        v38 = (unsigned int)v70;
                      }
                      v69[v38] = v25;
                      v21 = (unsigned int)(v70 + 1);
                      LODWORD(v70) = v70 + 1;
                    }
                    else
                    {
                      v66 = 128;
                      sub_C43690(&v65, v27, 0);
                      sub_C47360(&v65, &v61);
                      sub_C4A1D0(&v67, &v65, &v63);
                      v57 = v68;
                      if ( v68 > 0x40 )
                      {
                        v28 = -1;
                        if ( v57 - (unsigned int)sub_C444A0(&v67) <= 0x40 )
                          v28 = *v67;
                      }
                      else
                      {
                        v28 = (__int64)v67;
                      }
                      v29 = sub_BCB2E0(v12);
                      v30 = sub_ACD640(v29, v28, 0);
                      v33 = sub_B8C140((__int64)&v60, v30, v31, v32);
                      v34 = (unsigned int)v70;
                      if ( (unsigned __int64)(unsigned int)v70 + 1 > HIDWORD(v70) )
                      {
                        v58 = v33;
                        sub_C8D5F0(&v69, v71, (unsigned int)v70 + 1LL, 8);
                        v34 = (unsigned int)v70;
                        v33 = v58;
                      }
                      v69[v34] = v33;
                      LODWORD(v70) = v70 + 1;
                      if ( v68 > 0x40 && v67 )
                        j_j___libc_free_0_0(v67);
                      if ( v66 > 0x40 && v65 )
                        j_j___libc_free_0_0(v65);
                      v21 = (unsigned int)v70;
                    }
                    v23 += 2;
                  }
                }
              }
            }
            v22 = sub_B9C770(v12, v69, (__int64 *)v21, 0, 1);
            sub_B99FD0((__int64)a1, 2u, v22);
            if ( v64 > 0x40 && v63 )
              j_j___libc_free_0_0(v63);
            if ( v62 > 0x40 && v61 )
              j_j___libc_free_0_0(v61);
            if ( v69 != v71 )
              _libc_free(v69, 2);
          }
        }
      }
    }
  }
}
