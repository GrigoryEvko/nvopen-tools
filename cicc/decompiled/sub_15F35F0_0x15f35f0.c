// Function: sub_15F35F0
// Address: 0x15f35f0
//
void __fastcall sub_15F35F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  _BYTE *v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  _WORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  _WORD *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned int i; // r13d
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  _QWORD *v31; // rsi
  __int64 v32; // rdx
  unsigned int v33; // r15d
  __int64 v34; // r13
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  _QWORD *v44; // rsi
  unsigned int v45; // r13d
  int v46; // eax
  unsigned int v47; // edx
  unsigned int v48; // [rsp+10h] [rbp-C0h]
  __int64 v49; // [rsp+10h] [rbp-C0h]
  __int64 v50; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v51; // [rsp+30h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-98h]
  __int64 v53; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v54; // [rsp+48h] [rbp-88h]
  __int64 v55; // [rsp+50h] [rbp-80h] BYREF
  unsigned int v56; // [rsp+58h] [rbp-78h]
  __int64 *v57; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v58; // [rsp+68h] [rbp-68h]
  _QWORD *v59; // [rsp+70h] [rbp-60h] BYREF
  __int64 v60; // [rsp+78h] [rbp-58h]
  _QWORD v61[10]; // [rsp+80h] [rbp-50h] BYREF

  if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
  {
    v4 = sub_1625790(a1, 2);
    v5 = v4;
    if ( v4 )
    {
      v6 = -(__int64)*(unsigned int *)(v4 + 8);
      v7 = *(_BYTE **)(v5 + 8 * v6);
      if ( !*v7 )
      {
        if ( (v8 = sub_161E970(*(_QWORD *)(v5 + 8 * v6)), v9 == 14)
          && *(_QWORD *)v8 == 0x775F68636E617262LL
          && *(_DWORD *)(v8 + 8) == 1751607653
          && *(_WORD *)(v8 + 12) == 29556
          || (v10 = (_WORD *)sub_161E970(v7), v11 == 2) && *v10 == 20566 )
        {
          v50 = sub_16498A0(a1);
          v59 = v61;
          v60 = 0x300000000LL;
          v12 = *(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8));
          LODWORD(v60) = 1;
          v52 = 128;
          v61[0] = v12;
          sub_16A4EF0(&v51, a2, 0);
          v54 = 128;
          sub_16A4EF0(&v53, a3, 0);
          v13 = sub_161E970(v7);
          if ( v14 == 14
            && *(_QWORD *)v13 == 0x775F68636E617262LL
            && *(_DWORD *)(v13 + 8) == 1751607653
            && *(_WORD *)(v13 + 12) == 29556 )
          {
            v21 = *(unsigned int *)(v5 + 8);
            for ( i = 1; i < (unsigned int)v21; ++i )
            {
              v29 = *(_QWORD *)(v5 + 8 * (i - v21));
              if ( *(_BYTE *)v29 != 1 || (v30 = *(_QWORD *)(v29 + 136), *(_BYTE *)(v30 + 16) != 13) )
                sub_41A0AE();
              v31 = *(_QWORD **)(v30 + 24);
              if ( *(_DWORD *)(v30 + 32) > 0x40u )
                v31 = (_QWORD *)*v31;
              v56 = 128;
              sub_16A4EF0(&v55, v31, 0);
              sub_16A7C10(&v55, &v51);
              sub_16A9D70(&v57, &v55, &v53);
              v48 = v58;
              if ( v58 <= 0x40 )
              {
                v23 = (__int64)v57;
              }
              else
              {
                v23 = -1;
                if ( v48 - (unsigned int)sub_16A57B0(&v57) <= 0x40 )
                  v23 = *v57;
              }
              v24 = sub_16498A0(a1);
              v25 = sub_1643360(v24);
              v26 = sub_159C470(v25, v23, 0);
              v27 = sub_161BD20(&v50, v26);
              v28 = (unsigned int)v60;
              if ( (unsigned int)v60 >= HIDWORD(v60) )
              {
                v49 = v27;
                sub_16CD150(&v59, v61, 0, 8);
                v28 = (unsigned int)v60;
                v27 = v49;
              }
              v59[v28] = v27;
              LODWORD(v60) = v60 + 1;
              if ( v58 > 0x40 && v57 )
                j_j___libc_free_0_0(v57);
              if ( v56 > 0x40 && v55 )
                j_j___libc_free_0_0(v55);
              v21 = *(unsigned int *)(v5 + 8);
            }
          }
          else
          {
            v15 = (_WORD *)sub_161E970(v7);
            if ( v16 == 2 && *v15 == 20566 )
            {
              v32 = *(unsigned int *)(v5 + 8);
              if ( (unsigned int)v32 > 1 )
              {
                v33 = 1;
                do
                {
                  v40 = *(_QWORD *)(v5 + 8 * (v33 - v32));
                  v41 = (unsigned int)v60;
                  if ( (unsigned int)v60 >= HIDWORD(v60) )
                  {
                    sub_16CD150(&v59, v61, 0, 8);
                    v41 = (unsigned int)v60;
                  }
                  v59[v41] = v40;
                  LODWORD(v60) = v60 + 1;
                  v42 = *(_QWORD *)(v5 + 8 * (v33 + 1 - (unsigned __int64)*(unsigned int *)(v5 + 8)));
                  if ( *(_BYTE *)v42 != 1 || (v43 = *(_QWORD *)(v42 + 136), *(_BYTE *)(v43 + 16) != 13) )
                    BUG();
                  v44 = *(_QWORD **)(v43 + 24);
                  if ( *(_DWORD *)(v43 + 32) > 0x40u )
                    v44 = (_QWORD *)*v44;
                  v56 = 128;
                  sub_16A4EF0(&v55, v44, 0);
                  sub_16A7C10(&v55, &v51);
                  sub_16A9D70(&v57, &v55, &v53);
                  v45 = v58;
                  if ( v58 <= 0x40 )
                  {
                    v34 = (__int64)v57;
                  }
                  else
                  {
                    v46 = sub_16A57B0(&v57);
                    v47 = v45;
                    v34 = -1;
                    if ( v47 - v46 <= 0x40 )
                      v34 = *v57;
                  }
                  v35 = sub_16498A0(a1);
                  v36 = sub_1643360(v35);
                  v37 = sub_159C470(v36, v34, 0);
                  v38 = sub_161BD20(&v50, v37);
                  v39 = (unsigned int)v60;
                  if ( (unsigned int)v60 >= HIDWORD(v60) )
                  {
                    sub_16CD150(&v59, v61, 0, 8);
                    v39 = (unsigned int)v60;
                  }
                  v59[v39] = v38;
                  LODWORD(v60) = v60 + 1;
                  if ( v58 > 0x40 && v57 )
                    j_j___libc_free_0_0(v57);
                  if ( v56 > 0x40 && v55 )
                    j_j___libc_free_0_0(v55);
                  v32 = *(unsigned int *)(v5 + 8);
                  v33 += 2;
                }
                while ( v33 < (unsigned int)v32 );
              }
            }
          }
          v17 = (unsigned __int64)v59;
          v18 = (unsigned int)v60;
          v19 = sub_16498A0(a1);
          v20 = sub_1627350(v19, v17, v18, 0, 1);
          sub_1625C10(a1, 2, v20);
          if ( v54 > 0x40 && v53 )
            j_j___libc_free_0_0(v53);
          if ( v52 > 0x40 && v51 )
            j_j___libc_free_0_0(v51);
          if ( v59 != v61 )
            _libc_free((unsigned __int64)v59);
        }
      }
    }
  }
}
