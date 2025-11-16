// Function: sub_38A9450
// Address: 0x38a9450
//
__int64 __fastcall sub_38A9450(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r12
  int v7; // eax
  unsigned int v8; // r14d
  __int64 v9; // rbx
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned int v13; // r9d
  unsigned __int64 v15; // r13
  unsigned __int64 v16; // rsi
  char v17; // al
  __int64 *v18; // rdi
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  int v21; // eax
  unsigned __int64 v22; // rsi
  __int64 *v23; // rdi
  __int64 v24; // [rsp+0h] [rbp-D0h]
  __int64 v25; // [rsp+8h] [rbp-C8h]
  char v26; // [rsp+15h] [rbp-BBh]
  __int16 v27; // [rsp+16h] [rbp-BAh]
  __int64 v28; // [rsp+18h] [rbp-B8h]
  int v29; // [rsp+20h] [rbp-B0h]
  char *v32; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-98h]
  char **v34; // [rsp+40h] [rbp-90h] BYREF
  const char *v35; // [rsp+48h] [rbp-88h]
  __int16 v36; // [rsp+50h] [rbp-80h]
  __int64 v37[4]; // [rsp+60h] [rbp-70h] BYREF
  char *v38; // [rsp+80h] [rbp-50h] BYREF
  char **v39; // [rsp+88h] [rbp-48h]
  __int64 v40; // [rsp+90h] [rbp-40h]
  __int64 v41; // [rsp+98h] [rbp-38h]

  v6 = a1 + 8;
  v37[0] = 0;
  v37[1] = 0;
  v37[2] = 0x8000000000000000LL;
  v37[3] = 0x7FFFFFFFFFFFFFFFLL;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v7 = *(_DWORD *)(a1 + 64);
  if ( v7 == 13 )
  {
    v15 = *(_QWORD *)(a1 + 56);
    if ( !(unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
    {
LABEL_12:
      LOWORD(v40) = 259;
      v38 = "missing required field 'count'";
      return (unsigned int)sub_38814C0(v6, v15, (__int64)&v38);
    }
    return 1;
  }
  if ( v7 == 372 )
  {
    v8 = 0;
    v29 = 0;
    v27 = 0;
    v28 = 0;
    v25 = -1;
    v26 = 0;
    v24 = 0x7FFFFFFFFFFFFFFFLL;
    v9 = -1;
    while ( 1 )
    {
      if ( sub_2241AC0(a1 + 72, "count") )
      {
        if ( sub_2241AC0(a1 + 72, "lowerBound") )
        {
          v16 = *(_QWORD *)(a1 + 56);
          v34 = (char **)"invalid field '";
          v36 = 1027;
          v38 = (char *)&v34;
          LOWORD(v40) = 770;
          v35 = (const char *)(a1 + 72);
          v39 = (char **)"'";
          v17 = sub_38814C0(v6, v16, (__int64)&v38);
        }
        else
        {
          v17 = sub_388AAB0(a1, (__int64)"lowerBound", 10, (__int64)v37);
        }
      }
      else
      {
        v33 = 5;
        v32 = "count";
        if ( !(_BYTE)v8 )
        {
          v10 = sub_3887100(v6);
          v11 = (__int64)v32;
          v12 = v33;
          *(_DWORD *)(a1 + 64) = v10;
          if ( v10 == 390 )
          {
            v38 = (char *)v9;
            LOBYTE(v39) = v26;
            v40 = v25;
            v41 = v24;
            if ( (unsigned __int8)sub_388A880(a1, v11, v12, (__int64)&v38) )
              return 1;
            v8 = 1;
            v29 = 1;
            v9 = (__int64)v38;
            v26 = (char)v39;
            v25 = v40;
            v24 = v41;
          }
          else
          {
            v38 = (char *)v28;
            LOWORD(v39) = v27;
            if ( (unsigned __int8)sub_38A2910(a1, v11, v12, (__int64)&v38, a4, a5, a6) )
              return 1;
            v8 = 1;
            v29 = 2;
            v28 = (__int64)v38;
            v27 = (__int16)v39;
          }
          goto LABEL_16;
        }
        v38 = "field '";
        v39 = &v32;
        v36 = 770;
        v20 = *(_QWORD *)(a1 + 56);
        v34 = &v38;
        LOWORD(v40) = 1283;
        v35 = "' cannot be specified more than once";
        v17 = sub_38814C0(v6, v20, (__int64)&v34);
      }
      if ( v17 )
        return 1;
LABEL_16:
      if ( *(_DWORD *)(a1 + 64) != 4 )
        goto LABEL_17;
      v21 = sub_3887100(v6);
      *(_DWORD *)(a1 + 64) = v21;
      if ( v21 != 372 )
        goto LABEL_28;
    }
  }
  v29 = 0;
  v8 = 0;
  v9 = -1;
  v28 = 0;
LABEL_28:
  v22 = *(_QWORD *)(a1 + 56);
  LOWORD(v40) = 259;
  v38 = "expected field label here";
  if ( (unsigned __int8)sub_38814C0(v6, v22, (__int64)&v38) )
    return 1;
LABEL_17:
  v15 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_388AF10(a1, 13, "expected ')' here") )
    return 1;
  if ( !(_BYTE)v8 )
    goto LABEL_12;
  if ( v29 == 1 )
  {
    v23 = *(__int64 **)a1;
    if ( a3 )
      v19 = sub_15BB740(v23, v9, v37[0], 1u, 1);
    else
      v19 = sub_15BB740(v23, v9, v37[0], 0, 1);
    v13 = 0;
  }
  else
  {
    if ( v29 != 2 )
      return v8;
    v18 = *(__int64 **)a1;
    if ( a3 )
      v19 = sub_15BB200(v18, v28, v37[0], 1u, 1);
    else
      v19 = sub_15BB200(v18, v28, v37[0], 0, 1);
    v13 = 0;
  }
  *a2 = v19;
  return v13;
}
