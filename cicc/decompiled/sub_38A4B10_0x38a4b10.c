// Function: sub_38A4B10
// Address: 0x38a4b10
//
__int64 __fastcall sub_38A4B10(__int64 a1, __int64 *a2, char a3, double a4, double a5, double a6)
{
  __int64 v6; // r13
  int v8; // eax
  char v9; // al
  int v10; // eax
  unsigned __int64 v11; // r14
  unsigned int v12; // r15d
  __int64 *v13; // rdi
  __int64 v14; // rax
  int v16; // eax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rsi
  int v20; // eax
  __int64 v21; // rcx
  unsigned int v22; // r8d
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  unsigned int v26; // eax
  unsigned __int64 v27; // rsi
  int v29; // [rsp+18h] [rbp-D8h] BYREF
  char v30; // [rsp+1Ch] [rbp-D4h]
  __int64 v31; // [rsp+20h] [rbp-D0h] BYREF
  __int16 v32; // [rsp+28h] [rbp-C8h]
  char *v33; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-B8h]
  __int64 v35; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v36; // [rsp+48h] [rbp-A8h]
  __int64 v37; // [rsp+50h] [rbp-A0h]
  _QWORD v38[2]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v39; // [rsp+70h] [rbp-80h]
  char **v40; // [rsp+80h] [rbp-70h] BYREF
  const char *v41; // [rsp+88h] [rbp-68h]
  __int16 v42; // [rsp+90h] [rbp-60h]
  char *v43; // [rsp+A0h] [rbp-50h] BYREF
  char *v44; // [rsp+A8h] [rbp-48h]
  __int16 v45; // [rsp+B0h] [rbp-40h]

  v6 = a1 + 8;
  v29 = 0;
  v30 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 255;
  v31 = 0;
  v32 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v8 = *(_DWORD *)(a1 + 64);
  if ( v8 == 13 )
    goto LABEL_9;
  if ( v8 == 372 )
  {
    while ( sub_2241AC0(a1 + 72, "flags") )
    {
      if ( sub_2241AC0(a1 + 72, "cc") )
      {
        if ( sub_2241AC0(a1 + 72, "types") )
        {
          v19 = *(_QWORD *)(a1 + 56);
          v41 = (const char *)(a1 + 72);
          v45 = 770;
          v40 = (char **)"invalid field '";
          v42 = 1027;
          v43 = (char *)&v40;
          v44 = "'";
          v9 = sub_38814C0(v6, v19, (__int64)&v43);
        }
        else
        {
          v9 = sub_38A29E0(a1, (__int64)"types", 5, (__int64)&v31, a4, a5, a6);
        }
        goto LABEL_6;
      }
      v34 = 2;
      v33 = "cc";
      if ( (_BYTE)v36 )
      {
        v18 = *(_QWORD *)(a1 + 56);
        v43 = "field '";
        v44 = (char *)&v33;
        v40 = &v43;
        v45 = 1283;
        v41 = "' cannot be specified more than once";
        v42 = 770;
        v9 = sub_38814C0(v6, v18, (__int64)&v40);
        goto LABEL_6;
      }
      v20 = sub_3887100(v6);
      v23 = (__int64)v33;
      v24 = v34;
      *(_DWORD *)(a1 + 64) = v20;
      if ( v20 == 390 )
      {
        v9 = sub_3889300(a1, v23, v24, (__int64)&v35);
        goto LABEL_6;
      }
      if ( v20 != 382 )
      {
        v25 = *(_QWORD *)(a1 + 56);
        v45 = 259;
        v43 = "expected DWARF calling convention";
        v9 = sub_38814C0(v6, v25, (__int64)&v43);
        goto LABEL_6;
      }
      v26 = sub_14E8B30(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80), v24, v21, v22);
      if ( !v26 )
      {
        v41 = (const char *)(a1 + 72);
        v43 = "invalid DWARF calling convention";
        v44 = " '";
        v42 = 1026;
        v27 = *(_QWORD *)(a1 + 56);
        v40 = &v43;
        v39 = 770;
        v45 = 771;
        v38[0] = &v40;
        v38[1] = "'";
        v9 = sub_38814C0(v6, v27, (__int64)v38);
        goto LABEL_6;
      }
      LOBYTE(v36) = 1;
      v35 = v26;
      v10 = sub_3887100(v6);
      *(_DWORD *)(a1 + 64) = v10;
LABEL_8:
      if ( v10 != 4 )
        goto LABEL_9;
      v16 = sub_3887100(v6);
      *(_DWORD *)(a1 + 64) = v16;
      if ( v16 != 372 )
        goto LABEL_17;
    }
    v9 = sub_388BBA0(a1, (__int64)"flags", 5, (__int64)&v29);
LABEL_6:
    if ( v9 )
      return 1;
    v10 = *(_DWORD *)(a1 + 64);
    goto LABEL_8;
  }
LABEL_17:
  v17 = *(_QWORD *)(a1 + 56);
  v45 = 259;
  v43 = "expected field label here";
  if ( (unsigned __int8)sub_38814C0(v6, v17, (__int64)&v43) )
  {
    return 1;
  }
  else
  {
LABEL_9:
    v11 = *(_QWORD *)(a1 + 56);
    v12 = sub_388AF10(a1, 13, "expected ')' here");
    if ( !(_BYTE)v12 )
    {
      if ( (_BYTE)v32 )
      {
        v13 = *(__int64 **)a1;
        if ( a3 )
          v14 = sub_15BEF40(v13, v29, v35, v31, 1u, 1);
        else
          v14 = sub_15BEF40(v13, v29, v35, v31, 0, 1);
        *a2 = v14;
      }
      else
      {
        v45 = 259;
        v43 = "missing required field 'types'";
        return (unsigned int)sub_38814C0(v6, v11, (__int64)&v43);
      }
    }
  }
  return v12;
}
