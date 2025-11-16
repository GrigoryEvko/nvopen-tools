// Function: sub_1225840
// Address: 0x1225840
//
__int64 __fastcall sub_1225840(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  unsigned __int64 v4; // rsi
  int v5; // eax
  char v6; // al
  unsigned __int64 v7; // r14
  unsigned int v8; // r15d
  __int64 *v9; // rdi
  __int64 v10; // rax
  int v12; // eax
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // eax
  unsigned __int64 v16; // [rsp-10h] [rbp-140h]
  char v17; // [rsp+10h] [rbp-120h]
  char v18; // [rsp+18h] [rbp-118h]
  __int64 v21; // [rsp+30h] [rbp-100h] BYREF
  __int16 v22; // [rsp+38h] [rbp-F8h]
  unsigned __int64 v23; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v24; // [rsp+48h] [rbp-E8h]
  __int64 v25; // [rsp+50h] [rbp-E0h]
  char *v26; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v27; // [rsp+68h] [rbp-C8h]
  _QWORD v28[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v29; // [rsp+80h] [rbp-B0h]
  char *v30[2]; // [rsp+90h] [rbp-A0h] BYREF
  _QWORD v31[2]; // [rsp+A0h] [rbp-90h] BYREF
  __int16 v32; // [rsp+B0h] [rbp-80h]
  _BYTE *v33; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v34; // [rsp+C8h] [rbp-68h]
  _BYTE v35[32]; // [rsp+D0h] [rbp-60h] BYREF
  char v36; // [rsp+F0h] [rbp-40h]

  v3 = a1 + 176;
  v23 = 0;
  v24 = 0;
  v25 = 0xFFFF;
  v21 = 0;
  v22 = 256;
  v33 = v35;
  v34 = 0x400000000LL;
  v36 = 0;
  v4 = 12;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
  {
LABEL_14:
    v8 = 1;
    goto LABEL_15;
  }
  v5 = *(_DWORD *)(a1 + 240);
  if ( v5 != 13 )
  {
    if ( v5 == 507 )
    {
      do
      {
        if ( (unsigned int)sub_2241AC0(a1 + 248, "tag") )
        {
          if ( (unsigned int)sub_2241AC0(a1 + 248, "header") )
          {
            if ( (unsigned int)sub_2241AC0(a1 + 248, "operands") )
            {
              v28[0] = a1 + 248;
              v26 = "invalid field '";
              v30[0] = (char *)&v26;
              v29 = 1027;
              v31[0] = "'";
              v32 = 770;
              goto LABEL_33;
            }
            if ( v36 )
            {
              v4 = *(_QWORD *)(a1 + 232);
              v32 = 1283;
              v30[0] = "field '";
              v31[0] = "operands";
              v26 = (char *)v30;
              v31[1] = 8;
              v28[0] = "' cannot be specified more than once";
              v29 = 770;
              sub_11FD800(v3, v4, (__int64)&v26, 1);
              goto LABEL_14;
            }
            v12 = sub_1205200(v3);
            v4 = (unsigned __int64)&v26;
            v27 = 0x400000000LL;
            *(_DWORD *)(a1 + 240) = v12;
            v26 = (char *)v28;
            v6 = sub_1225600(a1, (__int64)&v26);
            if ( !v6 )
            {
              v30[1] = (char *)0x400000000LL;
              v30[0] = (char *)v31;
              if ( (_DWORD)v27 )
              {
                sub_12059F0((__int64)v30, &v26, (__int64)v31, v13, v14, (__int64)v30);
                v6 = 0;
              }
              v4 = (unsigned __int64)v30;
              v17 = v6;
              v36 = 1;
              sub_12059F0((__int64)&v33, v30, (__int64)v31, v13, v14, (__int64)v30);
              v6 = v17;
              if ( (_QWORD *)v30[0] != v31 )
              {
                _libc_free(v30[0], v30);
                v6 = v17;
              }
            }
            if ( v26 != (char *)v28 )
            {
              v18 = v6;
              _libc_free(v26, v4);
              v6 = v18;
            }
          }
          else
          {
            v4 = (unsigned __int64)"header";
            v6 = sub_120BB20(a1, "header", 6, (__int64)&v21);
          }
        }
        else
        {
          v4 = (unsigned __int64)"tag";
          v6 = sub_1208B00(a1, (__int64)"tag", 3, (__int64)&v23);
        }
        if ( v6 )
          goto LABEL_14;
        if ( *(_DWORD *)(a1 + 240) != 4 )
          goto LABEL_8;
        v15 = sub_1205200(v3);
        *(_DWORD *)(a1 + 240) = v15;
      }
      while ( v15 == 507 );
    }
    v30[0] = "expected field label here";
    v32 = 259;
LABEL_33:
    v4 = *(_QWORD *)(a1 + 232);
    sub_11FD800(v3, v4, (__int64)v30, 1);
    goto LABEL_14;
  }
LABEL_8:
  v4 = 13;
  v7 = *(_QWORD *)(a1 + 232);
  v8 = sub_120AFE0(a1, 13, "expected ')' here");
  if ( (_BYTE)v8 )
    goto LABEL_14;
  if ( !(_BYTE)v24 )
  {
    v4 = v7;
    v32 = 259;
    v30[0] = "missing required field 'tag'";
    sub_11FD800(v3, v7, (__int64)v30, 1);
    goto LABEL_14;
  }
  v4 = v23;
  v9 = *(__int64 **)a1;
  if ( a3 )
  {
    v10 = sub_B029A0(v9, v23, v21, (__int64)v33, (unsigned int)v34, 1u, 1);
    v4 = v16;
  }
  else
  {
    v10 = sub_B029A0(v9, v23, v21, (__int64)v33, (unsigned int)v34, 0, 1);
  }
  *a2 = v10;
LABEL_15:
  if ( v33 != v35 )
    _libc_free(v33, v4);
  return v8;
}
