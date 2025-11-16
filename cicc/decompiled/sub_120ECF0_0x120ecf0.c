// Function: sub_120ECF0
// Address: 0x120ecf0
//
__int64 __fastcall sub_120ECF0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v5; // eax
  char v6; // al
  int v7; // eax
  unsigned __int64 v8; // r15
  unsigned int v9; // r14d
  unsigned __int8 v10; // r15
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 v13; // r13
  unsigned int v14; // r8d
  __int64 v15; // rax
  bool v16; // cc
  const char *v17; // rax
  unsigned __int64 v19; // rsi
  int v20; // eax
  int v21; // eax
  unsigned __int64 v22; // rsi
  unsigned __int64 v23; // rsi
  char v24; // al
  __int16 v26; // [rsp+1Eh] [rbp-C2h] BYREF
  __int64 v27; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v28; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v29; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+38h] [rbp-A8h]
  char v31; // [rsp+40h] [rbp-A0h]
  const char *v32; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-88h]
  char v34; // [rsp+5Ch] [rbp-84h]
  const char *v35; // [rsp+60h] [rbp-80h]
  __int16 v36; // [rsp+70h] [rbp-70h]
  char *v37; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v38; // [rsp+88h] [rbp-58h]
  char v39; // [rsp+8Ch] [rbp-54h]
  char *v40; // [rsp+90h] [rbp-50h]
  __int64 v41; // [rsp+98h] [rbp-48h]
  __int16 v42; // [rsp+A0h] [rbp-40h]

  v3 = a1 + 176;
  v27 = 0;
  v28 = 256;
  v29 = 0;
  v30 = 1;
  v31 = 0;
  v26 = 0;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    goto LABEL_32;
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 == 13 )
  {
LABEL_9:
    v8 = *(_QWORD *)(a1 + 232);
    v9 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v9 )
      goto LABEL_32;
    if ( (_BYTE)v28 )
    {
      if ( v31 )
      {
        v10 = v26;
        v11 = v30;
        if ( !(_BYTE)v26 || BYTE4(v30) )
        {
          v33 = v30;
          if ( (unsigned int)v30 <= 0x40 )
            goto LABEL_17;
        }
        else
        {
          v12 = 1LL << ((unsigned __int8)v30 - 1);
          if ( (unsigned int)v30 <= 0x40 )
          {
            if ( (v12 & v29) == 0 )
            {
              v33 = v30;
LABEL_17:
              v32 = (const char *)v29;
              goto LABEL_18;
            }
LABEL_63:
            v23 = *(_QWORD *)(a1 + 232);
            v42 = 259;
            v9 = (unsigned __int8)v26;
            v37 = "unsigned enumerator with negative value";
            sub_11FD800(v3, v23, (__int64)&v37, 1);
            goto LABEL_33;
          }
          if ( (*(_QWORD *)(v29 + 8LL * ((unsigned int)(v30 - 1) >> 6)) & v12) != 0 )
            goto LABEL_63;
          v33 = v30;
        }
        sub_C43780((__int64)&v32, (const void **)&v29);
        v10 = v26;
        v11 = v33;
LABEL_18:
        v34 = BYTE4(v30);
        if ( v10 != 1 && BYTE4(v30) )
        {
          v22 = v29;
          if ( (unsigned int)v30 > 0x40 )
            v22 = *(_QWORD *)(v29 + 8LL * ((unsigned int)(v30 - 1) >> 6));
          v10 = 0;
          if ( (v22 & (1LL << ((unsigned __int8)v30 - 1))) != 0 )
          {
            sub_C449B0((__int64)&v37, (const void **)&v32, v11 + 1);
            if ( v33 > 0x40 && v32 )
              j_j___libc_free_0_0(v32);
            v10 = v26;
            v32 = v37;
            v11 = v38;
            v33 = v38;
          }
        }
        v38 = v11;
        v13 = v27;
        if ( a3 )
        {
          if ( v11 > 0x40 )
            sub_C43780((__int64)&v37, (const void **)&v32);
          else
            v37 = (char *)v32;
          v14 = 1;
        }
        else
        {
          if ( v11 > 0x40 )
            sub_C43780((__int64)&v37, (const void **)&v32);
          else
            v37 = (char *)v32;
          v14 = 0;
        }
        v15 = sub_B046D0(*(__int64 **)a1, (__int64)&v37, v10, v13, v14, 1);
        v16 = v38 <= 0x40;
        *a2 = v15;
        if ( !v16 && v37 )
          j_j___libc_free_0_0(v37);
        if ( v33 > 0x40 && v32 )
          j_j___libc_free_0_0(v32);
        goto LABEL_33;
      }
      HIBYTE(v42) = 1;
      v17 = "missing required field 'value'";
    }
    else
    {
      HIBYTE(v42) = 1;
      v17 = "missing required field 'name'";
    }
    v37 = (char *)v17;
    LOBYTE(v42) = 3;
    sub_11FD800(v3, v8, (__int64)&v37, 1);
    goto LABEL_32;
  }
  if ( v5 != 507 )
  {
LABEL_43:
    v37 = "expected field label here";
    v42 = 259;
    goto LABEL_52;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "name") )
    {
      v6 = sub_120BB20(a1, "name", 4, (__int64)&v27);
LABEL_6:
      if ( v6 )
        goto LABEL_32;
      v7 = *(_DWORD *)(a1 + 240);
      goto LABEL_8;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "value") )
      break;
    if ( v31 )
    {
      v37 = "field '";
      v40 = "value";
      v36 = 770;
      v19 = *(_QWORD *)(a1 + 232);
      v42 = 1283;
      v32 = (const char *)&v37;
      v41 = 5;
      v35 = "' cannot be specified more than once";
      sub_11FD800(v3, v19, (__int64)&v32, 1);
      goto LABEL_32;
    }
    v21 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v21;
    if ( v21 != 529 )
    {
      v37 = "expected integer";
      v42 = 259;
      goto LABEL_52;
    }
    v38 = *(_DWORD *)(a1 + 328);
    if ( v38 > 0x40 )
      sub_C43780((__int64)&v37, (const void **)(a1 + 320));
    else
      v37 = *(char **)(a1 + 320);
    v24 = *(_BYTE *)(a1 + 332);
    v31 = 1;
    v39 = v24;
    if ( (unsigned int)v30 > 0x40 && v29 )
    {
      j_j___libc_free_0_0(v29);
      v24 = v39;
    }
    BYTE4(v30) = v24;
    v29 = (unsigned __int64)v37;
    LODWORD(v30) = v38;
    v7 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v7;
LABEL_8:
    if ( v7 != 4 )
      goto LABEL_9;
    v20 = sub_1205200(v3);
    *(_DWORD *)(a1 + 240) = v20;
    if ( v20 != 507 )
      goto LABEL_43;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "isUnsigned") )
  {
    v6 = sub_1207D20(a1, (__int64)"isUnsigned", 10, (__int64)&v26);
    goto LABEL_6;
  }
  v35 = (const char *)(a1 + 248);
  v32 = "invalid field '";
  v36 = 1027;
  v37 = (char *)&v32;
  v40 = "'";
  v42 = 770;
LABEL_52:
  sub_11FD800(v3, *(_QWORD *)(a1 + 232), (__int64)&v37, 1);
LABEL_32:
  v9 = 1;
LABEL_33:
  if ( (unsigned int)v30 > 0x40 && v29 )
    j_j___libc_free_0_0(v29);
  return v9;
}
