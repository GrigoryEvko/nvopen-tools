// Function: sub_120F790
// Address: 0x120f790
//
__int64 __fastcall sub_120F790(__int64 a1, const __m128i **a2, char a3)
{
  __int64 v6; // r13
  int v7; // eax
  char v8; // al
  int v9; // eax
  unsigned int v10; // r14d
  char v11; // al
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 *v14; // rdi
  __m128i v15; // xmm1
  const __m128i *v16; // rax
  const char *v17; // rax
  int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // rsi
  int v22; // edx
  unsigned __int64 v23; // rsi
  int v24; // eax
  unsigned __int64 v25; // rsi
  __m128i v26; // xmm2
  unsigned __int64 v27; // rsi
  __int32 v28; // [rsp+0h] [rbp-120h]
  char v29; // [rsp+7h] [rbp-119h]
  unsigned __int64 v30; // [rsp+8h] [rbp-118h]
  unsigned __int8 v31; // [rsp+8h] [rbp-118h]
  __int64 v32; // [rsp+18h] [rbp-108h]
  __int64 v33; // [rsp+20h] [rbp-100h] BYREF
  __int16 v34; // [rsp+28h] [rbp-F8h]
  __int64 v35; // [rsp+30h] [rbp-F0h] BYREF
  __int16 v36; // [rsp+38h] [rbp-E8h]
  __int64 v37; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v38; // [rsp+48h] [rbp-D8h]
  __int64 v39; // [rsp+50h] [rbp-D0h] BYREF
  __int16 v40; // [rsp+58h] [rbp-C8h]
  _QWORD v41[4]; // [rsp+60h] [rbp-C0h] BYREF
  __int16 v42; // [rsp+80h] [rbp-A0h]
  __m128i v43; // [rsp+90h] [rbp-90h] BYREF
  const char *v44; // [rsp+A0h] [rbp-80h]
  __int16 v45; // [rsp+B0h] [rbp-70h]
  __m128i v46; // [rsp+C0h] [rbp-60h] BYREF
  char *v47; // [rsp+D0h] [rbp-50h]
  __int64 v48; // [rsp+D8h] [rbp-48h]
  __int16 v49; // [rsp+E0h] [rbp-40h]

  v38 = 256;
  v6 = a1 + 176;
  v33 = 0;
  v34 = 256;
  v35 = 0;
  v36 = 256;
  v37 = 0;
  v39 = 0;
  v40 = 256;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' here") )
    return 1;
  v7 = *(_DWORD *)(a1 + 240);
  if ( v7 == 13 )
  {
    v29 = 0;
    v28 = 1;
LABEL_9:
    v30 = *(_QWORD *)(a1 + 232);
    v10 = sub_120AFE0(a1, 13, "expected ')' here");
    if ( (_BYTE)v10 )
      return 1;
    if ( (_BYTE)v34 )
    {
      if ( (_BYTE)v36 )
      {
        v44 = 0;
        v43 = 0;
        if ( v29 )
        {
          if ( (_BYTE)v38 )
          {
            v12 = v37;
            v11 = 1;
            goto LABEL_15;
          }
        }
        else if ( !(_BYTE)v38 )
        {
          v28 = 0;
          v11 = 0;
          v12 = 0;
LABEL_15:
          v13 = 0;
          if ( (_BYTE)v40 )
            v13 = v39;
          v14 = *(__int64 **)a1;
          if ( a3 )
          {
            v43.m128i_i64[1] = v12;
            LOBYTE(v44) = v11;
            v43.m128i_i32[0] = v28;
            v15 = _mm_loadu_si128(&v43);
            v47 = (char *)v44;
            v46 = v15;
            v16 = sub_B07920(v14, v33, v35, v13, 1u, 1, (const __m128i)0LL, v15.m128i_i64[0]);
          }
          else
          {
            LOBYTE(v44) = v11;
            v43.m128i_i64[1] = v12;
            v43.m128i_i32[0] = v28;
            v26 = _mm_loadu_si128(&v43);
            v47 = (char *)v44;
            v46 = v26;
            v16 = sub_B07920(v14, v33, v35, v13, 0, 1, (const __m128i)0LL, v26.m128i_i64[0]);
          }
          *a2 = v16;
          return v10;
        }
        v25 = *(_QWORD *)(a1 + 232);
        v31 = v36;
        v46.m128i_i64[0] = (__int64)"'checksumkind' and 'checksum' must be provided together";
        v49 = 259;
        sub_11FD800(v6, v25, (__int64)&v46, 1);
        return v31;
      }
      HIBYTE(v49) = 1;
      v17 = "missing required field 'directory'";
    }
    else
    {
      HIBYTE(v49) = 1;
      v17 = "missing required field 'filename'";
    }
    v46.m128i_i64[0] = (__int64)v17;
    LOBYTE(v49) = 3;
    sub_11FD800(v6, v30, (__int64)&v46, 1);
    return 1;
  }
  v29 = 0;
  v28 = 1;
  if ( v7 != 507 )
  {
LABEL_35:
    v46.m128i_i64[0] = (__int64)"expected field label here";
    v49 = 259;
    goto LABEL_36;
  }
  while ( 1 )
  {
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "filename") )
    {
      v8 = sub_120BB20(a1, "filename", 8, (__int64)&v33);
LABEL_6:
      if ( v8 )
        return 1;
      v9 = *(_DWORD *)(a1 + 240);
      goto LABEL_8;
    }
    if ( !(unsigned int)sub_2241AC0(a1 + 248, "directory") )
    {
      v8 = sub_120BB20(a1, "directory", 9, (__int64)&v35);
      goto LABEL_6;
    }
    if ( (unsigned int)sub_2241AC0(a1 + 248, "checksumkind") )
      break;
    if ( v29 )
    {
      v27 = *(_QWORD *)(a1 + 232);
      v46.m128i_i64[0] = (__int64)"field '";
      v47 = "checksumkind";
      v43.m128i_i64[0] = (__int64)&v46;
      v49 = 1283;
      v48 = 12;
      v44 = "' cannot be specified more than once";
      v45 = 770;
      sub_11FD800(v6, v27, (__int64)&v43, 1);
      return 1;
    }
    v19 = sub_1205200(v6);
    v20 = *(_QWORD *)(a1 + 248);
    v21 = *(_QWORD *)(a1 + 256);
    *(_DWORD *)(a1 + 240) = v19;
    v32 = sub_AF2F90(v20, v21, v22);
    if ( *(_DWORD *)(a1 + 240) != 524 || !BYTE4(v32) )
    {
      v44 = (const char *)(a1 + 248);
      v46.m128i_i64[0] = (__int64)"invalid checksum kind";
      v47 = " '";
      v43.m128i_i64[0] = (__int64)&v46;
      v45 = 1026;
      v23 = *(_QWORD *)(a1 + 232);
      v49 = 771;
      v41[0] = &v43;
      v42 = 770;
      v41[2] = "'";
      sub_11FD800(v6, v23, (__int64)v41, 1);
      return 1;
    }
    v28 = v32;
    v9 = sub_1205200(v6);
    v29 = 1;
    *(_DWORD *)(a1 + 240) = v9;
LABEL_8:
    if ( v9 != 4 )
      goto LABEL_9;
    v24 = sub_1205200(v6);
    *(_DWORD *)(a1 + 240) = v24;
    if ( v24 != 507 )
      goto LABEL_35;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "checksum") )
  {
    v8 = sub_120BB20(a1, "checksum", 8, (__int64)&v37);
    goto LABEL_6;
  }
  if ( !(unsigned int)sub_2241AC0(a1 + 248, "source") )
  {
    v8 = sub_120BB20(a1, "source", 6, (__int64)&v39);
    goto LABEL_6;
  }
  v44 = (const char *)(a1 + 248);
  v43.m128i_i64[0] = (__int64)"invalid field '";
  v45 = 1027;
  v46.m128i_i64[0] = (__int64)&v43;
  v47 = "'";
  v49 = 770;
LABEL_36:
  sub_11FD800(v6, *(_QWORD *)(a1 + 232), (__int64)&v46, 1);
  return 1;
}
