// Function: sub_29E2B40
// Address: 0x29e2b40
//
void __fastcall sub_29E2B40(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // r8
  __int64 *v9; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 *v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdx
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 *v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned __int8 *v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rsi
  __int64 v37; // r12
  __int64 v38; // rsi
  unsigned __int8 *v39; // rsi
  __int64 v40; // rax
  __int64 v41; // rbx
  _QWORD *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // [rsp+8h] [rbp-E8h]
  _QWORD *v45; // [rsp+10h] [rbp-E0h]
  __int64 v46; // [rsp+18h] [rbp-D8h]
  __int64 *v47; // [rsp+28h] [rbp-C8h]
  __int64 v49; // [rsp+40h] [rbp-B0h]
  __int64 *v50; // [rsp+48h] [rbp-A8h]
  __int64 v51; // [rsp+48h] [rbp-A8h]
  _QWORD *v52; // [rsp+48h] [rbp-A8h]
  __int64 *v53; // [rsp+50h] [rbp-A0h]
  __int64 v54; // [rsp+50h] [rbp-A0h]
  __int64 v55; // [rsp+50h] [rbp-A0h]
  char v57; // [rsp+5Fh] [rbp-91h]
  _QWORD *v58; // [rsp+60h] [rbp-90h] BYREF
  unsigned __int8 *v59; // [rsp+68h] [rbp-88h] BYREF
  unsigned __int8 *v60; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int8 *v61; // [rsp+78h] [rbp-78h] BYREF
  __int64 v62[4]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v63; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v64; // [rsp+A8h] [rbp-48h]
  __int64 v65; // [rsp+B0h] [rbp-40h]
  unsigned int v66; // [rsp+B8h] [rbp-38h]

  if ( *(_QWORD *)(a3 + 48) )
  {
    v46 = a2;
    v47 = (__int64 *)sub_B2BE50(a1);
    v6 = sub_B10CD0(a3 + 48);
    v58 = (_QWORD *)v6;
    v7 = *(_BYTE *)(v6 - 16);
    if ( (v7 & 2) != 0 )
    {
      if ( *(_DWORD *)(v6 - 24) == 2 )
        v8 = *(_QWORD *)(*(_QWORD *)(v6 - 32) + 8LL);
      else
        v8 = 0;
      v9 = *(__int64 **)(v6 - 32);
    }
    else
    {
      v35 = v6 - 16;
      v8 = 0;
      if ( ((*(_WORD *)(v6 - 16) >> 6) & 0xF) == 2 )
        v8 = *(_QWORD *)(v35 - 8LL * ((v7 >> 2) & 0xF) + 8);
      v9 = (__int64 *)(v35 - 8LL * ((v7 >> 2) & 0xF));
    }
    v58 = sub_B01860(v47, *(_DWORD *)(v6 + 4), *(unsigned __int16 *)(v6 + 2), *v9, v8, 0, 1u, 1);
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v57 = sub_B2D620(a1, "no-inline-line-tables", 0x15u);
    v44 = a1 + 72;
    if ( a2 != a1 + 72 )
    {
      do
      {
        if ( !v46 )
          BUG();
        v49 = *(_QWORD *)(v46 + 32);
        if ( v46 + 24 == v49 )
          goto LABEL_54;
        do
        {
          v10 = 0;
          v62[2] = (__int64)&v63;
          if ( v49 )
            v10 = v49 - 24;
          v62[0] = (__int64)v47;
          v62[1] = (__int64)&v58;
          sub_AE8EA0(v10, (__int64 (__fastcall *)(__int64))sub_29E18A0, (__int64)v62);
          if ( !v57 )
          {
            v11 = *(_QWORD *)(v10 + 48);
            v59 = (unsigned __int8 *)v11;
            if ( v11 )
            {
              sub_B96E90((__int64)&v59, v11, 1);
              if ( v59 )
              {
                v12 = sub_BD5C60(v10);
                v13 = (__int64)v58;
                v14 = (__int64 *)v12;
                v61 = v59;
                if ( v59 )
                {
                  v45 = v58;
                  v50 = (__int64 *)v12;
                  sub_B96E90((__int64)&v61, (__int64)v59, 1);
                  v13 = (__int64)v45;
                  v14 = v50;
                }
                sub_29E11F0(&v60, (__int64)&v61, v13, v14, (__int64)&v63);
                if ( v61 )
                  sub_B91220((__int64)&v61, (__int64)v61);
                v61 = v60;
                if ( v60 )
                {
                  sub_B96E90((__int64)&v61, (__int64)v60, 1);
                  v15 = *(_QWORD *)(v10 + 48);
                  v16 = v10 + 48;
                  if ( v15 )
                    goto LABEL_20;
LABEL_21:
                  v17 = v61;
                  *(_QWORD *)(v10 + 48) = v61;
                  if ( v17 )
                    sub_B976B0((__int64)&v61, v17, v16);
                  if ( v60 )
                    sub_B91220((__int64)&v60, (__int64)v60);
                }
                else
                {
                  v15 = *(_QWORD *)(v10 + 48);
                  v16 = v10 + 48;
                  if ( v15 )
                  {
LABEL_20:
                    v51 = v16;
                    sub_B91220(v16, v15);
                    v16 = v51;
                    goto LABEL_21;
                  }
                }
                if ( v59 )
                  sub_B91220((__int64)&v59, (__int64)v59);
                goto LABEL_27;
              }
            }
            if ( a4 )
              goto LABEL_27;
          }
          if ( *(_BYTE *)v10 == 60 )
          {
            if ( **(_BYTE **)(v10 - 32) <= 0x15u && (*(_BYTE *)(v10 + 2) & 0x40) == 0 )
              goto LABEL_27;
          }
          else if ( *(_BYTE *)v10 == 85 )
          {
            v40 = *(_QWORD *)(v10 - 32);
            if ( v40 )
            {
              if ( !*(_BYTE *)v40
                && *(_QWORD *)(v40 + 24) == *(_QWORD *)(v10 + 80)
                && (*(_BYTE *)(v40 + 33) & 0x20) != 0
                && *(_DWORD *)(v40 + 36) == 291 )
              {
                goto LABEL_27;
              }
            }
          }
          v36 = *(_QWORD *)(a3 + 48);
          v61 = (unsigned __int8 *)v36;
          if ( v36 )
          {
            v37 = v10 + 48;
            sub_B96E90((__int64)&v61, v36, 1);
            v38 = *(_QWORD *)(v10 + 48);
            if ( !v38 )
              goto LABEL_71;
          }
          else
          {
            v38 = *(_QWORD *)(v10 + 48);
            v37 = v10 + 48;
            if ( !v38 )
              goto LABEL_27;
          }
          sub_B91220(v37, v38);
LABEL_71:
          v39 = v61;
          *(_QWORD *)(v10 + 48) = v61;
          if ( v39 )
            sub_B976B0((__int64)&v61, v39, v37);
LABEL_27:
          v18 = *(_QWORD *)(v10 + 64);
          if ( v18 )
          {
            v19 = sub_B14240(v18);
            v21 = v20;
            v22 = (_QWORD *)v19;
            if ( v20 != (_QWORD *)v19 )
            {
              while ( 1 )
              {
                if ( !v57 )
                {
                  v23 = v22[3];
                  v60 = (unsigned __int8 *)v23;
                  if ( v23 )
                    sub_B96E90((__int64)&v60, v23, 1);
                  v24 = sub_B14180(v22[2]);
                  v25 = sub_AA48A0(v24);
                  v26 = (__int64)v58;
                  v27 = (__int64 *)v25;
                  v62[0] = (__int64)v60;
                  if ( v60 )
                  {
                    v52 = v58;
                    v53 = (__int64 *)v25;
                    sub_B96E90((__int64)v62, (__int64)v60, 1);
                    v26 = (__int64)v52;
                    v27 = v53;
                  }
                  sub_29E11F0(&v61, (__int64)v62, v26, v27, (__int64)&v63);
                  if ( v62[0] )
                    sub_B91220((__int64)v62, v62[0]);
                  v62[0] = (__int64)v61;
                  if ( v61 )
                  {
                    sub_B96E90((__int64)v62, (__int64)v61, 1);
                    v28 = v22[3];
                    v29 = (__int64)(v22 + 3);
                    if ( v28 )
                      goto LABEL_38;
LABEL_39:
                    v30 = (unsigned __int8 *)v62[0];
                    v22[3] = v62[0];
                    if ( v30 )
                      sub_B976B0((__int64)v62, v30, v29);
                    if ( v61 )
                      sub_B91220((__int64)&v61, (__int64)v61);
                  }
                  else
                  {
                    v28 = v22[3];
                    v29 = (__int64)(v22 + 3);
                    if ( v28 )
                    {
LABEL_38:
                      v54 = v29;
                      sub_B91220(v29, v28);
                      v29 = v54;
                      goto LABEL_39;
                    }
                  }
                  if ( v60 )
                    sub_B91220((__int64)&v60, (__int64)v60);
                  goto LABEL_45;
                }
                v31 = *(_QWORD *)(a3 + 48);
                v62[0] = v31;
                if ( !v31 )
                  break;
                sub_B96E90((__int64)v62, v31, 1);
                v32 = v22[3];
                v33 = (__int64)(v22 + 3);
                if ( v32 )
                  goto LABEL_49;
LABEL_50:
                v34 = (unsigned __int8 *)v62[0];
                v22[3] = v62[0];
                if ( v34 )
                {
                  sub_B976B0((__int64)v62, v34, v33);
                  v22 = (_QWORD *)v22[1];
                  if ( v22 == v21 )
                    goto LABEL_52;
                }
                else
                {
LABEL_45:
                  v22 = (_QWORD *)v22[1];
                  if ( v22 == v21 )
                    goto LABEL_52;
                }
              }
              v32 = v22[3];
              v33 = (__int64)(v22 + 3);
              if ( !v32 )
                goto LABEL_45;
LABEL_49:
              v55 = v33;
              sub_B91220(v33, v32);
              v33 = v55;
              goto LABEL_50;
            }
          }
LABEL_52:
          v49 = *(_QWORD *)(v49 + 8);
        }
        while ( v46 + 24 != v49 );
        if ( v57 )
        {
          v41 = *(_QWORD *)(v46 + 32);
          while ( v49 != v41 )
          {
            if ( !v41 )
              BUG();
            v42 = (_QWORD *)(v41 - 24);
            if ( *(_BYTE *)(v41 - 24) == 85
              && (v43 = *(_QWORD *)(v41 - 56)) != 0
              && !*(_BYTE *)v43
              && *(_QWORD *)(v43 + 24) == *(_QWORD *)(v41 + 56)
              && (*(_BYTE *)(v43 + 33) & 0x20) != 0
              && (unsigned int)(*(_DWORD *)(v43 + 36) - 68) <= 3 )
            {
              v41 = sub_B43D60(v42);
            }
            else
            {
              sub_B44570((__int64)v42);
              v41 = *(_QWORD *)(v41 + 8);
            }
          }
        }
LABEL_54:
        v46 = *(_QWORD *)(v46 + 8);
      }
      while ( v46 != v44 );
    }
    sub_C7D6A0(v64, 16LL * v66, 8);
  }
}
