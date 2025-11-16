// Function: sub_388D380
// Address: 0x388d380
//
__int64 __fastcall sub_388D380(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r12
  int v4; // eax
  char v5; // r14
  __int64 v6; // r15
  char v7; // al
  int v8; // eax
  unsigned int v9; // r13d
  __int64 *v10; // rdi
  __int64 v11; // rax
  unsigned __int64 v13; // rsi
  unsigned __int64 v14; // rsi
  const char *v15; // rax
  int v16; // eax
  unsigned __int64 v17; // rsi
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-E0h]
  char v22; // [rsp+Fh] [rbp-D1h]
  unsigned __int64 v23; // [rsp+10h] [rbp-D0h]
  __int64 v24; // [rsp+18h] [rbp-C8h]
  char *v25; // [rsp+20h] [rbp-C0h]
  char v26; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v27; // [rsp+28h] [rbp-B8h]
  int v28; // [rsp+30h] [rbp-B0h]
  __int16 v31; // [rsp+4Eh] [rbp-92h] BYREF
  __int64 v32; // [rsp+50h] [rbp-90h] BYREF
  __int16 v33; // [rsp+58h] [rbp-88h]
  char *v34; // [rsp+60h] [rbp-80h] BYREF
  __int64 v35; // [rsp+68h] [rbp-78h]
  char **v36; // [rsp+70h] [rbp-70h] BYREF
  const char *v37; // [rsp+78h] [rbp-68h]
  __int16 v38; // [rsp+80h] [rbp-60h]
  char *v39; // [rsp+90h] [rbp-50h] BYREF
  char **v40; // [rsp+98h] [rbp-48h]
  unsigned __int64 v41; // [rsp+A0h] [rbp-40h]
  __int64 v42; // [rsp+A8h] [rbp-38h]

  v3 = a1 + 8;
  v32 = 0;
  v33 = 256;
  v31 = 0;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v4 = *(_DWORD *)(a1 + 64);
  if ( v4 == 13 )
  {
    v28 = 0;
    v5 = 0;
    v6 = 0;
    v25 = 0;
  }
  else
  {
    if ( v4 == 372 )
    {
      v5 = 0;
      v6 = 0;
      v21 = 0x7FFFFFFFFFFFFFFFLL;
      v28 = 0;
      v24 = -1;
      v26 = 0;
      v25 = 0;
      v23 = 0x8000000000000000LL;
      v22 = 0;
      while ( 1 )
      {
        if ( sub_2241AC0(a1 + 72, "name") )
        {
          if ( sub_2241AC0(a1 + 72, "value") )
          {
            if ( sub_2241AC0(a1 + 72, "isUnsigned") )
            {
              v17 = *(_QWORD *)(a1 + 56);
              v37 = (const char *)(a1 + 72);
              LOWORD(v41) = 770;
              v36 = (char **)"invalid field '";
              v38 = 1027;
              v39 = (char *)&v36;
              v40 = (char **)"'";
              v7 = sub_38814C0(v3, v17, (__int64)&v39);
            }
            else
            {
              v7 = sub_3887760(a1, (__int64)"isUnsigned", 10, (__int64)&v31);
            }
          }
          else
          {
            v35 = 5;
            v34 = "value";
            if ( !v5 )
            {
              v8 = sub_3887100(v3);
              v18 = (__int64)v34;
              v19 = v35;
              *(_DWORD *)(a1 + 64) = v8;
              if ( v8 == 390 )
              {
                if ( *(_BYTE *)(a1 + 164) )
                {
                  v39 = v25;
                  LOBYTE(v40) = v26;
                  v41 = v24;
                  if ( (unsigned __int8)sub_3889300(a1, v18, v19, (__int64)&v39) )
                    return 1;
                  v5 = 1;
                  v28 = 2;
                  v25 = v39;
                  v26 = (char)v40;
                  v24 = v41;
                  v8 = *(_DWORD *)(a1 + 64);
                }
                else
                {
                  v39 = (char *)v6;
                  LOBYTE(v40) = v22;
                  v41 = v23;
                  v42 = v21;
                  if ( (unsigned __int8)sub_388A880(a1, v18, v19, (__int64)&v39) )
                    return 1;
                  v6 = (__int64)v39;
                  v5 = 1;
                  v28 = 1;
                  v22 = (char)v40;
                  v23 = v41;
                  v21 = v42;
                  v8 = *(_DWORD *)(a1 + 64);
                }
              }
              goto LABEL_9;
            }
            v39 = "field '";
            v40 = &v34;
            v38 = 770;
            v14 = *(_QWORD *)(a1 + 56);
            v36 = &v39;
            LOWORD(v41) = 1283;
            v37 = "' cannot be specified more than once";
            v7 = sub_38814C0(v3, v14, (__int64)&v36);
          }
        }
        else
        {
          v7 = sub_388B8F0(a1, (__int64)"name", 4, (__int64)&v32);
        }
        if ( v7 )
          return 1;
        v8 = *(_DWORD *)(a1 + 64);
LABEL_9:
        if ( v8 != 4 )
          goto LABEL_10;
        v16 = sub_3887100(v3);
        *(_DWORD *)(a1 + 64) = v16;
        if ( v16 != 372 )
          goto LABEL_21;
      }
    }
    v28 = 0;
    v5 = 0;
    v6 = 0;
    v25 = 0;
LABEL_21:
    v13 = *(_QWORD *)(a1 + 56);
    LOWORD(v41) = 259;
    v39 = "expected field label here";
    if ( (unsigned __int8)sub_38814C0(v3, v13, (__int64)&v39) )
      return 1;
  }
LABEL_10:
  v27 = *(_QWORD *)(a1 + 56);
  v9 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v9 )
  {
    if ( (_BYTE)v33 )
    {
      if ( v5 )
      {
        if ( (_BYTE)v31 )
        {
          if ( v28 == 1 )
          {
            v20 = *(_QWORD *)(a1 + 56);
            LOWORD(v41) = 259;
            v39 = "unsigned enumerator with negative value";
            return (unsigned int)sub_38814C0(v3, v20, (__int64)&v39);
          }
        }
        else if ( v28 == 1 )
        {
LABEL_15:
          v10 = *(__int64 **)a1;
          if ( a3 )
            v11 = sub_15BC290(v10, v6, v31, v32, 1u, 1);
          else
            v11 = sub_15BC290(v10, v6, v31, v32, 0, 1);
          *a2 = v11;
          return v9;
        }
        v6 = (__int64)v25;
        goto LABEL_15;
      }
      BYTE1(v41) = 1;
      v15 = "missing required field 'value'";
    }
    else
    {
      BYTE1(v41) = 1;
      v15 = "missing required field 'name'";
    }
    v39 = (char *)v15;
    LOBYTE(v41) = 3;
    return (unsigned int)sub_38814C0(v3, v27, (__int64)&v39);
  }
  return v9;
}
