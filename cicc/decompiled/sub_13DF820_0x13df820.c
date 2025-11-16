// Function: sub_13DF820
// Address: 0x13df820
//
__int64 __fastcall sub_13DF820(__int64 a1, __int64 a2, _QWORD *a3, int a4)
{
  __int64 *v4; // r14
  __int64 v6; // r12
  unsigned __int8 v8; // al
  __int64 v9; // r15
  char v11; // al
  char v12; // al
  char v13; // al
  unsigned int v14; // eax
  unsigned __int64 **v15; // r13
  unsigned __int64 *v16; // r10
  unsigned int v17; // r12d
  _QWORD *v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 *v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 *v26; // rax
  _BYTE *v27; // rsi
  __int64 v28; // rdx
  int v29; // eax
  _BYTE *v30; // rsi
  int v31; // eax
  __int64 v32; // rdx
  void *v33; // rax
  int v34; // eax
  __int64 v35; // [rsp+0h] [rbp-100h]
  unsigned int v36; // [rsp+8h] [rbp-F8h]
  unsigned int v37; // [rsp+10h] [rbp-F0h]
  unsigned int v38; // [rsp+10h] [rbp-F0h]
  __int64 v39; // [rsp+10h] [rbp-F0h]
  unsigned int v40; // [rsp+18h] [rbp-E8h]
  unsigned int v41; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v42; // [rsp+18h] [rbp-E8h]
  __int64 v43; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD *v44; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+48h] [rbp-B8h] BYREF
  __int64 v46; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v47; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v48[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v49; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v50; // [rsp+78h] [rbp-88h] BYREF
  _QWORD *v51; // [rsp+80h] [rbp-80h] BYREF
  int v52; // [rsp+88h] [rbp-78h]
  _QWORD *v53; // [rsp+90h] [rbp-70h] BYREF
  __int64 v54; // [rsp+98h] [rbp-68h]
  __int64 v55[2]; // [rsp+A0h] [rbp-60h] BYREF
  __int64 *v56; // [rsp+B0h] [rbp-50h] BYREF
  __int64 *v57; // [rsp+B8h] [rbp-48h]
  __int64 v58[8]; // [rsp+C0h] [rbp-40h] BYREF

  v4 = (__int64 *)a1;
  v6 = a2;
  v8 = *(_BYTE *)(a2 + 16);
  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    if ( v8 > 0x10u )
    {
      v8 = *(_BYTE *)(a1 + 16);
      v4 = (__int64 *)a2;
      v6 = a1;
    }
    else
    {
      v9 = sub_14D6F90(26, a1, a2, *a3);
      if ( v9 )
        return v9;
      v8 = *(_BYTE *)(a2 + 16);
    }
  }
  if ( v8 != 9 )
  {
    if ( (__int64 *)v6 == v4 )
      return v6;
    if ( !sub_13CD190(v6) )
    {
      v9 = (__int64)v4;
      if ( (unsigned __int8)sub_13CC520(v6) )
        return v9;
      v53 = (_QWORD *)v6;
      if ( !sub_13D1F50((__int64 *)&v53, (__int64)v4) )
      {
        v56 = v4;
        if ( !sub_13D1F50((__int64 *)&v56, v6) )
        {
          v11 = *((_BYTE *)v4 + 16);
          if ( v11 != 51 )
          {
            if ( v11 == 5 && *((_WORD *)v4 + 9) == 27 )
            {
              v19 = v4[-3 * (*((_DWORD *)v4 + 5) & 0xFFFFFFF)];
              if ( v6 == v19 )
              {
                if ( v19 )
                  return v6;
              }
              v20 = v4[3 * (1LL - (*((_DWORD *)v4 + 5) & 0xFFFFFFF))];
              if ( v20 )
              {
                if ( v6 == v20 )
                  return v6;
              }
            }
LABEL_15:
            v12 = *(_BYTE *)(v6 + 16);
            if ( v12 == 51 )
            {
              v23 = *(__int64 **)(v6 - 48);
              if ( v23 && v4 == v23 )
                return (__int64)v4;
              v24 = *(__int64 **)(v6 - 24);
              if ( v4 == v24 )
              {
                v9 = (__int64)v4;
                if ( v24 )
                  return v9;
              }
            }
            else if ( v12 == 5 && *(_WORD *)(v6 + 18) == 27 )
            {
              v25 = *(__int64 **)(v6 - 24LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF));
              if ( v4 == v25 )
              {
                if ( v25 )
                  return (__int64)v4;
              }
              v26 = *(__int64 **)(v6 + 24 * (1LL - (*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
              if ( v4 == v26 )
              {
                if ( v26 )
                  return (__int64)v4;
              }
            }
            v56 = (__int64 *)&v44;
            if ( !(unsigned __int8)sub_13D2630(&v56, (_BYTE *)v6) )
              goto LABEL_25;
            v49 = &v43;
            v50 = &v45;
            v13 = *((_BYTE *)v4 + 16);
            if ( v13 == 47 )
            {
              if ( !*(v4 - 6) )
              {
LABEL_22:
                v49 = &v43;
                v50 = &v45;
                if ( v13 == 48 )
                {
                  if ( !*(v4 - 6) )
                    goto LABEL_25;
                  v30 = (_BYTE *)*(v4 - 3);
                  v43 = *(v4 - 6);
                  if ( !(unsigned __int8)sub_13D2630(&v50, v30) )
                    goto LABEL_25;
                }
                else
                {
                  if ( v13 != 5 )
                    goto LABEL_25;
                  if ( *((_WORD *)v4 + 9) != 24 )
                    goto LABEL_25;
                  v32 = *((_DWORD *)v4 + 5) & 0xFFFFFFF;
                  if ( !v4[-3 * v32] )
                    goto LABEL_25;
                  v43 = v4[-3 * v32];
                  if ( !(unsigned __int8)sub_13D7780(&v50, (_BYTE *)v4[3 * (1 - v32)]) )
                    goto LABEL_25;
                }
                sub_13A38D0((__int64)&v51, (__int64)v44);
                sub_13D0570((__int64)&v51);
                v31 = v52;
                v52 = 0;
                LODWORD(v54) = v31;
                v39 = v45;
                v53 = v51;
                sub_13A38D0((__int64)&v56, (__int64)&v53);
                sub_16A7E20(&v56, v39);
                LOBYTE(v39) = sub_13D01C0((__int64)&v56);
                sub_135E100((__int64 *)&v56);
                sub_135E100((__int64 *)&v53);
                v9 = (__int64)v4;
                sub_135E100((__int64 *)&v51);
                if ( (_BYTE)v39 )
                  return v9;
LABEL_25:
                v54 = v6;
                if ( sub_13D52E0((__int64)&v53, (__int64)v4) || (v57 = v4, sub_13D52E0((__int64)&v56, v6)) )
                {
                  v9 = (__int64)v4;
                  if ( (unsigned __int8)sub_14BDFF0((_DWORD)v4, *a3, 1, 0, a3[3], a3[4], a3[2]) )
                    return v9;
                  v9 = v6;
                  if ( (unsigned __int8)sub_14BDFF0(v6, *a3, 1, 0, a3[3], a3[4], a3[2]) )
                    return v9;
                }
                v9 = sub_13D4EE0((__int64)v4, v6, 1);
                if ( v9 )
                  return v9;
                v9 = (__int64)sub_13DDF20(26, (unsigned __int8 *)v4, (unsigned __int8 *)v6, a3, a4);
                if ( v9 )
                  return v9;
                v9 = (__int64)sub_13DF2B0(26, (unsigned __int8 *)v4, (unsigned __int8 *)v6, 27, a3, a4);
                if ( v9 )
                  return v9;
                v9 = (__int64)sub_13DF2B0(26, (unsigned __int8 *)v4, (unsigned __int8 *)v6, 28, a3, a4);
                if ( v9 )
                  return v9;
                if ( *((_BYTE *)v4 + 16) == 79 || *(_BYTE *)(v6 + 16) == 79 )
                {
                  v33 = sub_13DF4D0(0x1Au, (unsigned __int8 *)v4, (unsigned __int8 *)v6, a3, a4);
                  if ( v33 )
                    return (__int64)v33;
                }
                if ( *((_BYTE *)v4 + 16) == 77 || *(_BYTE *)(v6 + 16) == 77 )
                {
                  v33 = sub_13DF6F0(26, (unsigned __int8 *)v4, (unsigned __int8 *)v6, a3, a4);
                  if ( v33 )
                    return (__int64)v33;
                }
                v53 = &v44;
                if ( !(unsigned __int8)sub_13D2630(&v53, (_BYTE *)v6) )
                  return v9;
                v56 = &v43;
                v57 = &v45;
                v58[0] = (__int64)&v47;
                v58[1] = (__int64)&v46;
                if ( !(unsigned __int8)sub_13D55B0(&v56, (__int64)v4) )
                  return v9;
                v14 = sub_16431D0(*v4);
                v15 = (unsigned __int64 **)v45;
                LODWORD(v16) = v14;
                v17 = v14;
                v37 = *(_DWORD *)(v45 + 8);
                if ( v37 > 0x40 )
                {
                  v42 = v14;
                  v34 = sub_16A57B0(v45);
                  LODWORD(v16) = v42;
                  if ( v37 - v34 <= 0x40 && v42 > **v15 )
                    v16 = (unsigned __int64 *)**v15;
                }
                else if ( (unsigned __int64)v14 > *(_QWORD *)v45 )
                {
                  v16 = *(unsigned __int64 **)v45;
                }
                v40 = (unsigned int)v16;
                sub_14C2530((unsigned int)&v53, v46, *a3, 0, a3[3], a3[4], a3[2], 0);
                v36 = v17 - sub_13D05A0((__int64)&v53);
                v38 = v40;
                if ( v40 >= v36 )
                {
                  sub_14C2530((unsigned int)&v56, v43, *a3, 0, a3[3], a3[4], a3[2], 0);
                  v41 = v17 - sub_13D05A0((__int64)&v56);
                  sub_13D0120((__int64)v48, v17, v36);
                  sub_13D0120((__int64)&v51, v17, v41);
                  sub_13A38D0((__int64)&v49, (__int64)&v51);
                  sub_13CC1B0((__int64)&v49, v38);
                  sub_135E100((__int64 *)&v51);
                  v18 = v44;
                  if ( sub_13D0550((__int64)v48, v44) && !sub_13D0530((__int64)&v49, v18) )
                  {
                    v9 = v46;
                    goto LABEL_46;
                  }
                  if ( sub_13D0550((__int64)&v49, v18) && !sub_13D0530((__int64)v48, v18) )
                  {
                    v9 = v47;
LABEL_46:
                    sub_135E100((__int64 *)&v49);
                    sub_135E100(v48);
                    sub_135E100(v58);
                    sub_135E100((__int64 *)&v56);
                    sub_135E100(v55);
                    sub_135E100((__int64 *)&v53);
                    return v9;
                  }
                  sub_135E100((__int64 *)&v49);
                  sub_135E100(v48);
                  sub_135E100(v58);
                  sub_135E100((__int64 *)&v56);
                }
                sub_135E100(v55);
                sub_135E100((__int64 *)&v53);
                return v9;
              }
              v27 = (_BYTE *)*(v4 - 3);
              v43 = *(v4 - 6);
              if ( !(unsigned __int8)sub_13D2630(&v50, v27) )
              {
LABEL_74:
                v13 = *((_BYTE *)v4 + 16);
                goto LABEL_22;
              }
            }
            else
            {
              if ( v13 != 5 )
                goto LABEL_22;
              if ( *((_WORD *)v4 + 9) != 23 )
                goto LABEL_22;
              v28 = *((_DWORD *)v4 + 5) & 0xFFFFFFF;
              if ( !v4[-3 * v28] )
                goto LABEL_22;
              v43 = v4[-3 * v28];
              if ( !(unsigned __int8)sub_13D7780(&v50, (_BYTE *)v4[3 * (1 - v28)]) )
                goto LABEL_74;
            }
            sub_13A38D0((__int64)&v51, (__int64)v44);
            sub_13D0570((__int64)&v51);
            v29 = v52;
            v52 = 0;
            LODWORD(v54) = v29;
            v35 = v45;
            v53 = v51;
            sub_13A38D0((__int64)&v56, (__int64)&v53);
            sub_16A81B0(&v56, v35);
            LOBYTE(v35) = sub_13D01C0((__int64)&v56);
            sub_135E100((__int64 *)&v56);
            sub_135E100((__int64 *)&v53);
            v9 = (__int64)v4;
            sub_135E100((__int64 *)&v51);
            if ( (_BYTE)v35 )
              return v9;
            goto LABEL_74;
          }
          v21 = *(v4 - 6);
          if ( !v21 || v6 != v21 )
          {
            v22 = *(v4 - 3);
            if ( v6 == v22 )
            {
              v9 = v6;
              if ( v22 )
                return v9;
            }
            goto LABEL_15;
          }
          return v6;
        }
      }
    }
  }
  return sub_15A06D0(*v4);
}
