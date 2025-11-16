// Function: sub_13D7FB0
// Address: 0x13d7fb0
//
_QWORD *__fastcall sub_13D7FB0(__int64 a1, __int64 a2, _QWORD *a3, unsigned int a4)
{
  _QWORD *v4; // r14
  _QWORD *v6; // r12
  unsigned __int8 v8; // al
  __int64 v9; // r15
  _QWORD *v11; // rax
  char v12; // al
  char v13; // al
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  int v17; // eax
  char v18; // al
  __int64 v19; // rax
  char v20; // [rsp+18h] [rbp-B8h]
  _QWORD *v21; // [rsp+38h] [rbp-98h] BYREF
  _QWORD *v22; // [rsp+40h] [rbp-90h] BYREF
  __int64 v23; // [rsp+48h] [rbp-88h] BYREF
  _QWORD *v24; // [rsp+50h] [rbp-80h] BYREF
  _QWORD **v25; // [rsp+58h] [rbp-78h]
  _QWORD *v26; // [rsp+60h] [rbp-70h] BYREF
  __int64 *v27; // [rsp+68h] [rbp-68h]
  _QWORD *v28; // [rsp+80h] [rbp-50h] BYREF
  _QWORD **v29; // [rsp+88h] [rbp-48h]
  _QWORD *v30; // [rsp+90h] [rbp-40h]

  v4 = (_QWORD *)a1;
  v6 = (_QWORD *)a2;
  v8 = *(_BYTE *)(a2 + 16);
  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    if ( v8 > 0x10u )
    {
      v8 = *(_BYTE *)(a1 + 16);
      v4 = (_QWORD *)a2;
      v6 = (_QWORD *)a1;
    }
    else
    {
      v9 = sub_14D6F90(27, a1, a2, *a3);
      if ( v9 )
        return (_QWORD *)v9;
      v8 = *(_BYTE *)(a2 + 16);
    }
  }
  if ( v8 != 9 && !(unsigned __int8)sub_13CC520((__int64)v6) )
  {
    if ( v6 == v4 )
      return v6;
    v9 = (__int64)v4;
    if ( sub_13CD190((__int64)v6) )
      return (_QWORD *)v9;
    v26 = v6;
    if ( !sub_13D1F50((__int64 *)&v26, (__int64)v4) )
    {
      v28 = v4;
      if ( !sub_13D1F50((__int64 *)&v28, (__int64)v6) )
      {
        v28 = v6;
        if ( (unsigned __int8)sub_13D4800(&v28, (__int64)v4) )
          return v6;
        v28 = v4;
        if ( (unsigned __int8)sub_13D4800(&v28, (__int64)v6) )
          return (_QWORD *)v9;
        v28 = v6;
        if ( sub_13D5AC0(&v28, (__int64)v4) )
          return (_QWORD *)sub_15A04A0(*v6);
        v28 = v6;
        if ( !sub_13D5AC0(&v28, (__int64)v6) )
        {
          v12 = *((_BYTE *)v6 + 16);
          if ( v12 == 52 )
          {
            if ( !*(v6 - 6) )
              goto LABEL_27;
            v21 = (_QWORD *)*(v6 - 6);
            v11 = (_QWORD *)*(v6 - 3);
            if ( !v11 )
              goto LABEL_27;
          }
          else
          {
            if ( v12 != 5 )
              goto LABEL_27;
            if ( *((_WORD *)v6 + 9) != 28 )
              goto LABEL_27;
            v15 = *((_DWORD *)v6 + 5) & 0xFFFFFFF;
            if ( !v6[-3 * v15] )
              goto LABEL_27;
            v21 = (_QWORD *)v6[-3 * v15];
            v11 = (_QWORD *)v6[3 * (1 - v15)];
            if ( !v11 )
              goto LABEL_27;
          }
          v22 = v11;
          v26 = v21;
          v27 = v11;
          if ( sub_13D7B90((__int64 *)&v26, (__int64)v4) )
            return v6;
          v28 = v21;
          v30 = v22;
          if ( sub_13D7CA0((__int64 *)&v28, (__int64)v4) )
            return v6;
LABEL_27:
          v13 = *((_BYTE *)v4 + 16);
          if ( v13 == 52 )
          {
            if ( !*(v4 - 6) )
              goto LABEL_30;
            v21 = (_QWORD *)*(v4 - 6);
            v14 = (_QWORD *)*(v4 - 3);
            if ( !v14 )
              goto LABEL_30;
          }
          else
          {
            if ( v13 != 5 )
              goto LABEL_30;
            if ( *((_WORD *)v4 + 9) != 28 )
              goto LABEL_30;
            v16 = *((_DWORD *)v4 + 5) & 0xFFFFFFF;
            if ( !v4[-3 * v16] )
              goto LABEL_30;
            v21 = (_QWORD *)v4[-3 * v16];
            v14 = (_QWORD *)v4[3 * (1 - v16)];
            if ( !v14 )
              goto LABEL_30;
          }
          v22 = v14;
          v26 = v21;
          v27 = v14;
          if ( sub_13D7B90((__int64 *)&v26, (__int64)v6) )
            return v4;
          v28 = v21;
          v30 = v22;
          if ( sub_13D7CA0((__int64 *)&v28, (__int64)v6) )
            return v4;
LABEL_30:
          v24 = &v21;
          v25 = &v22;
          if ( !(unsigned __int8)sub_13D5EF0(&v24, (__int64)v4)
            || (v26 = v21, v27 = v22, !sub_13D7DA0((__int64 *)&v26, (__int64)v6))
            && (v28 = v21, v30 = v22, !sub_13D7EB0((__int64 *)&v28, (__int64)v6)) )
          {
            v24 = &v21;
            v25 = &v22;
            if ( !(unsigned __int8)sub_13D5EF0(&v24, (__int64)v6)
              || (v26 = v21, v27 = v22, !sub_13D7DA0((__int64 *)&v26, (__int64)v4))
              && (v28 = v21, v30 = v22, !sub_13D7EB0((__int64 *)&v28, (__int64)v4)) )
            {
              v9 = sub_13D4EE0((__int64)v4, (__int64)v6, 0);
              if ( v9 )
                return (_QWORD *)v9;
              v9 = sub_13DDF20(27, v4, v6, a3, a4);
              if ( v9 )
                return (_QWORD *)v9;
              v9 = sub_13DF2B0(27, v4, v6, 26, a3, a4);
              if ( v9 )
                return (_QWORD *)v9;
              if ( *((_BYTE *)v4 + 16) == 79 || *((_BYTE *)v6 + 16) == 79 )
              {
                v19 = sub_13DF4D0(27, v4, v6, a3, a4);
                if ( v19 )
                  return (_QWORD *)v19;
              }
              v26 = &v21;
              v27 = &v23;
              if ( !(unsigned __int8)sub_13D5F90(&v26, (__int64)v4) )
                goto LABEL_43;
              v28 = &v22;
              v29 = &v24;
              if ( !(unsigned __int8)sub_13D5F90(&v28, (__int64)v6) )
                goto LABEL_43;
              sub_13A38D0((__int64)&v26, (__int64)v24);
              sub_13D0570((__int64)&v26);
              v17 = (int)v27;
              LODWORD(v27) = 0;
              LODWORD(v29) = v17;
              v28 = v26;
              v18 = *(_DWORD *)(v23 + 8) <= 0x40u ? *(_QWORD *)v23 == (_QWORD)v26 : sub_16A5220(v23, &v28);
              v20 = v18;
              sub_135E100((__int64 *)&v28);
              sub_135E100((__int64 *)&v26);
              if ( !v20 )
                goto LABEL_43;
              if ( sub_13CFFB0((__int64)v24) )
              {
                v28 = v22;
                v29 = &v26;
                if ( (unsigned __int8)sub_13D60C0((__int64)&v28, (__int64)v21) )
                {
                  if ( (unsigned __int8)sub_14C1670((_DWORD)v26, (_DWORD)v24, *a3, 0, a3[3], a3[4], a3[2]) )
                    return v21;
                }
              }
              if ( !sub_13CFFB0(v23)
                || (v28 = v21, v29 = &v26, !(unsigned __int8)sub_13D60C0((__int64)&v28, (__int64)v22))
                || !(unsigned __int8)sub_14C1670((_DWORD)v26, v23, *a3, 0, a3[3], a3[4], a3[2]) )
              {
LABEL_43:
                if ( *((_BYTE *)v4 + 16) == 77 || *((_BYTE *)v6 + 16) == 77 )
                  return (_QWORD *)sub_13DF6F0(27, v4, v6, a3, a4);
                return (_QWORD *)v9;
              }
              return v22;
            }
            return v4;
          }
          return v6;
        }
      }
    }
  }
  return (_QWORD *)sub_15A04A0(*v4);
}
