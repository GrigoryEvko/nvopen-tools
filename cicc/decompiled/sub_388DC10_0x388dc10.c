// Function: sub_388DC10
// Address: 0x388dc10
//
__int64 __fastcall sub_388DC10(__int64 a1, __int64 *a2, char a3)
{
  __int64 v3; // r13
  int v6; // eax
  char v7; // al
  int v8; // eax
  unsigned int v9; // r14d
  __int64 *v10; // rdi
  __int64 v11; // rax
  int v13; // eax
  unsigned __int64 v14; // rsi
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  const char *v19; // rax
  unsigned __int64 v20; // rsi
  unsigned __int64 v21; // rsi
  unsigned __int64 v22; // rsi
  __int64 v23; // [rsp+8h] [rbp-108h]
  __int32 v24; // [rsp+10h] [rbp-100h]
  char v25; // [rsp+17h] [rbp-F9h]
  unsigned __int64 v26; // [rsp+18h] [rbp-F8h]
  __int32 v27; // [rsp+28h] [rbp-E8h] BYREF
  char v28; // [rsp+2Ch] [rbp-E4h]
  __int64 v29; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v30; // [rsp+38h] [rbp-D8h]
  __int64 v31; // [rsp+40h] [rbp-D0h] BYREF
  __int16 v32; // [rsp+48h] [rbp-C8h]
  __int64 v33; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v34; // [rsp+58h] [rbp-B8h]
  __m128i *v35; // [rsp+60h] [rbp-B0h] BYREF
  __int16 v36; // [rsp+68h] [rbp-A8h]
  _QWORD v37[2]; // [rsp+70h] [rbp-A0h] BYREF
  _QWORD v38[2]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v39; // [rsp+90h] [rbp-80h]
  __m128i *v40; // [rsp+A0h] [rbp-70h] BYREF
  const char *v41; // [rsp+A8h] [rbp-68h]
  __int16 v42; // [rsp+B0h] [rbp-60h]
  __m128i v43; // [rsp+C0h] [rbp-50h] BYREF
  __int16 v44; // [rsp+D0h] [rbp-40h]

  v3 = a1 + 8;
  v30 = 256;
  v34 = 256;
  v29 = 0;
  v31 = 0;
  v32 = 256;
  v33 = 0;
  v35 = 0;
  v36 = 256;
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
    return 1;
  v6 = *(_DWORD *)(a1 + 64);
  if ( v6 == 13 )
  {
    v25 = 0;
    v24 = 1;
  }
  else
  {
    v25 = 0;
    v24 = 1;
    if ( v6 == 372 )
    {
      while ( sub_2241AC0(a1 + 72, "filename") )
      {
        if ( !sub_2241AC0(a1 + 72, "directory") )
        {
          v7 = sub_388B8F0(a1, (__int64)"directory", 9, (__int64)&v31);
          goto LABEL_6;
        }
        if ( sub_2241AC0(a1 + 72, "checksumkind") )
        {
          if ( sub_2241AC0(a1 + 72, "checksum") )
          {
            if ( sub_2241AC0(a1 + 72, "source") )
            {
              v22 = *(_QWORD *)(a1 + 56);
              v41 = (const char *)(a1 + 72);
              v40 = (__m128i *)"invalid field '";
              v43.m128i_i64[0] = (__int64)&v40;
              v42 = 1027;
              v43.m128i_i64[1] = (__int64)"'";
              v44 = 770;
              v7 = sub_38814C0(v3, v22, (__int64)&v43);
            }
            else
            {
              v7 = sub_388B8F0(a1, (__int64)"source", 6, (__int64)&v35);
            }
          }
          else
          {
            v7 = sub_388B8F0(a1, (__int64)"checksum", 8, (__int64)&v33);
          }
          goto LABEL_6;
        }
        v37[1] = 12;
        v37[0] = "checksumkind";
        if ( v25 )
        {
          v21 = *(_QWORD *)(a1 + 56);
          v44 = 1283;
          v43.m128i_i64[0] = (__int64)"field '";
          v43.m128i_i64[1] = (__int64)v37;
          v40 = &v43;
          v41 = "' cannot be specified more than once";
          v42 = 770;
          v7 = sub_38814C0(v3, v21, (__int64)&v40);
          goto LABEL_6;
        }
        v15 = sub_3887100(v3);
        v16 = *(_QWORD *)(a1 + 72);
        v17 = *(_QWORD *)(a1 + 80);
        *(_DWORD *)(a1 + 64) = v15;
        sub_15B0D60((__int64)&v27, v16, v17);
        if ( *(_DWORD *)(a1 + 64) != 387 || !v28 )
        {
          v18 = *(_QWORD *)(a1 + 56);
          v43.m128i_i64[0] = (__int64)"invalid checksum kind";
          v43.m128i_i64[1] = (__int64)" '";
          v40 = &v43;
          v38[0] = &v40;
          v38[1] = "'";
          v44 = 771;
          v41 = (const char *)(a1 + 72);
          v42 = 1026;
          v39 = 770;
          v7 = sub_38814C0(v3, v18, (__int64)v38);
          goto LABEL_6;
        }
        v24 = v27;
        v8 = sub_3887100(v3);
        v25 = 1;
        *(_DWORD *)(a1 + 64) = v8;
LABEL_8:
        if ( v8 != 4 )
          goto LABEL_9;
        v13 = sub_3887100(v3);
        *(_DWORD *)(a1 + 64) = v13;
        if ( v13 != 372 )
          goto LABEL_24;
      }
      v7 = sub_388B8F0(a1, (__int64)"filename", 8, (__int64)&v29);
LABEL_6:
      if ( v7 )
        return 1;
      v8 = *(_DWORD *)(a1 + 64);
      goto LABEL_8;
    }
LABEL_24:
    v14 = *(_QWORD *)(a1 + 56);
    v44 = 259;
    v43.m128i_i64[0] = (__int64)"expected field label here";
    if ( (unsigned __int8)sub_38814C0(v3, v14, (__int64)&v43) )
      return 1;
  }
LABEL_9:
  v26 = *(_QWORD *)(a1 + 56);
  v9 = sub_388AF10(a1, 13, "expected ')' here");
  if ( !(_BYTE)v9 )
  {
    if ( (_BYTE)v30 )
    {
      if ( (_BYTE)v32 )
      {
        if ( v25 )
        {
          if ( (_BYTE)v34 )
          {
            v23 = v33;
            goto LABEL_15;
          }
        }
        else if ( !(_BYTE)v34 )
        {
LABEL_15:
          v10 = *(__int64 **)a1;
          if ( (_BYTE)v36 )
          {
            if ( !a3 )
            {
              LOBYTE(v41) = 1;
              v40 = v35;
              if ( (_BYTE)v34 )
              {
LABEL_18:
                LOBYTE(v44) = 1;
                v43.m128i_i32[0] = v24;
                v43.m128i_i64[1] = v23;
LABEL_19:
                v11 = sub_15BF650(v10, v29, v31, &v43, (__int64)&v40, 0, 1);
LABEL_20:
                *a2 = v11;
                return v9;
              }
              goto LABEL_52;
            }
            LOBYTE(v41) = 1;
            v40 = v35;
            if ( (_BYTE)v34 )
            {
LABEL_47:
              LOBYTE(v44) = 1;
              v43.m128i_i32[0] = v24;
              v43.m128i_i64[1] = v23;
LABEL_48:
              v11 = sub_15BF650(v10, v29, v31, &v43, (__int64)&v40, 1u, 1);
              goto LABEL_20;
            }
          }
          else
          {
            if ( !a3 )
            {
              if ( (_BYTE)v34 )
              {
                LOBYTE(v41) = 0;
                goto LABEL_18;
              }
              LOBYTE(v41) = 0;
LABEL_52:
              LOBYTE(v44) = 0;
              goto LABEL_19;
            }
            LOBYTE(v41) = 0;
            if ( (_BYTE)v34 )
              goto LABEL_47;
          }
          LOBYTE(v44) = 0;
          goto LABEL_48;
        }
        v20 = *(_QWORD *)(a1 + 56);
        v44 = 259;
        v43.m128i_i64[0] = (__int64)"'checksumkind' and 'checksum' must be provided together";
        return (unsigned int)sub_38814C0(v3, v20, (__int64)&v43);
      }
      HIBYTE(v44) = 1;
      v19 = "missing required field 'directory'";
    }
    else
    {
      HIBYTE(v44) = 1;
      v19 = "missing required field 'filename'";
    }
    v43.m128i_i64[0] = (__int64)v19;
    LOBYTE(v44) = 3;
    return (unsigned int)sub_38814C0(v3, v26, (__int64)&v43);
  }
  return v9;
}
