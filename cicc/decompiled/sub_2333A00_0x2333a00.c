// Function: sub_2333A00
// Address: 0x2333a00
//
__int64 __fastcall sub_2333A00(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v3; // r13
  bool v4; // al
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rcx
  char v10; // r13
  bool v12; // al
  bool v13; // al
  bool v14; // al
  bool v15; // al
  unsigned int v16; // eax
  unsigned int v17; // ebx
  __int64 v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned int v21; // eax
  unsigned int v22; // ebx
  __int64 v23; // rdx
  __int64 v24; // r15
  unsigned __int64 v25; // rax
  char v26; // [rsp+15h] [rbp-FBh]
  char v27; // [rsp+16h] [rbp-FAh]
  char v28; // [rsp+17h] [rbp-F9h]
  __int64 v29; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v30; // [rsp+20h] [rbp-F0h]
  __m128i v31; // [rsp+28h] [rbp-E8h]
  char v32; // [rsp+38h] [rbp-D8h]
  char v33; // [rsp+39h] [rbp-D7h]
  char v34; // [rsp+3Ah] [rbp-D6h]
  char v35; // [rsp+3Bh] [rbp-D5h]
  char v36; // [rsp+3Ch] [rbp-D4h]
  char v37; // [rsp+3Dh] [rbp-D3h]
  char v38; // [rsp+3Eh] [rbp-D2h]
  char v39; // [rsp+3Fh] [rbp-D1h]
  __int64 v40; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int64 v41; // [rsp+48h] [rbp-C8h]
  __int64 v42; // [rsp+58h] [rbp-B8h] BYREF
  __m128i v43; // [rsp+60h] [rbp-B0h] BYREF
  unsigned __int64 v44; // [rsp+70h] [rbp-A0h] BYREF
  unsigned int v45; // [rsp+78h] [rbp-98h]
  unsigned __int64 v46[4]; // [rsp+80h] [rbp-90h] BYREF
  const char *v47; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v48; // [rsp+A8h] [rbp-68h]
  _QWORD *v49; // [rsp+B0h] [rbp-60h]
  __int64 v50; // [rsp+B8h] [rbp-58h]
  char v51; // [rsp+C0h] [rbp-50h]
  void *v52; // [rsp+C8h] [rbp-48h] BYREF
  __m128i *v53; // [rsp+D0h] [rbp-40h]
  _QWORD v54[7]; // [rsp+D8h] [rbp-38h] BYREF

  v3 = a1;
  v40 = a2;
  v41 = a3;
  v34 = 0;
  v36 = 1;
  v26 = 0;
  v32 = 0;
  v35 = 0;
  v37 = 1;
  v33 = 0;
  v27 = 1;
  LODWORD(v29) = 1;
  v39 = 1;
  v28 = 0;
  v38 = 0;
  if ( !a3 )
  {
LABEL_20:
    *(_BYTE *)(v3 + 13) = 0;
    *(_QWORD *)(v3 + 16) = 0;
    *(_DWORD *)v3 = v29;
    *(_BYTE *)(v3 + 4) = v33;
    *(_BYTE *)(v3 + 5) = v35;
    *(_BYTE *)(v3 + 6) = v38;
    *(_BYTE *)(v3 + 7) = v36;
    *(_BYTE *)(v3 + 8) = v34;
    *(_BYTE *)(v3 + 9) = v32;
    *(_BYTE *)(v3 + 10) = v26;
    *(_BYTE *)(v3 + 11) = v37;
    *(_BYTE *)(v3 + 12) = v27;
    *(_BYTE *)(v3 + 14) = v39;
    *(_BYTE *)(v3 + 15) = v28;
    *(_BYTE *)(v3 + 24) = *(_BYTE *)(v3 + 24) & 0xFC | 2;
    return v3;
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v43 = 0u;
      LOBYTE(v47) = 59;
      v6 = sub_C931B0(&v40, &v47, 1u, 0);
      if ( v6 == -1 )
      {
        v8 = v40;
        v6 = v41;
        v5 = 0;
        v9 = 0;
      }
      else
      {
        v7 = v6 + 1;
        v8 = v40;
        if ( v6 + 1 > v41 )
        {
          v7 = v41;
          v5 = 0;
        }
        else
        {
          v5 = v41 - v7;
        }
        v9 = v40 + v7;
        if ( v6 > v41 )
          v6 = v41;
      }
      v43.m128i_i64[0] = v8;
      v10 = 1;
      v43.m128i_i64[1] = v6;
      v40 = v9;
      v41 = v5;
      if ( v6 <= 2 )
      {
LABEL_3:
        if ( v6 == 19
          && !(*(_QWORD *)v8 ^ 0x2D64726177726F66LL | *(_QWORD *)(v8 + 8) ^ 0x632D686374697773LL)
          && *(_WORD *)(v8 + 16) == 28271
          && *(_BYTE *)(v8 + 18) == 100 )
        {
          v33 = v10;
          goto LABEL_7;
        }
        goto LABEL_5;
      }
      if ( *(_WORD *)v8 == 28526 && *(_BYTE *)(v8 + 2) == 45 )
      {
        v6 -= 3LL;
        v8 += 3;
        v10 = 0;
        v43.m128i_i64[0] = v8;
        v43.m128i_i64[1] = v6;
      }
      else
      {
        v10 = 1;
      }
      if ( v6 != 16 )
        break;
      if ( *(_QWORD *)v8 ^ 0x74616C7563657073LL | *(_QWORD *)(v8 + 8) ^ 0x736B636F6C622D65LL )
      {
        if ( !(*(_QWORD *)v8 ^ 0x742D686374697773LL | *(_QWORD *)(v8 + 8) ^ 0x70756B6F6F6C2D6FLL) )
        {
          v38 = v10;
          goto LABEL_7;
        }
        goto LABEL_5;
      }
      v39 = v10;
      if ( !v5 )
      {
LABEL_19:
        v3 = a1;
        goto LABEL_20;
      }
    }
    if ( v6 != 20 )
      goto LABEL_3;
    if ( !(*(_QWORD *)v8 ^ 0x7966696C706D6973LL | *(_QWORD *)(v8 + 8) ^ 0x72622D646E6F632DLL)
      && *(_DWORD *)(v8 + 16) == 1751346785 )
    {
      v37 = v10;
      goto LABEL_7;
    }
    if ( !(*(_QWORD *)v8 ^ 0x722D686374697773LL | *(_QWORD *)(v8 + 8) ^ 0x2D6F742D65676E61LL)
      && *(_DWORD *)(v8 + 16) == 1886217065 )
    {
      v35 = v10;
      goto LABEL_7;
    }
LABEL_5:
    v30 = v5;
    v31 = v43;
    v4 = sub_9691B0((const void *)v43.m128i_i64[0], v43.m128i_u64[1], "keep-loops", 10);
    v5 = v30;
    if ( !v4 )
      break;
    v36 = v10;
LABEL_7:
    if ( !v5 )
      goto LABEL_19;
  }
  v12 = sub_9691B0((const void *)v31.m128i_i64[0], v31.m128i_u64[1], "hoist-common-insts", 18);
  v5 = v30;
  if ( v12 )
  {
    v34 = v10;
    goto LABEL_7;
  }
  v13 = sub_9691B0((const void *)v31.m128i_i64[0], v31.m128i_u64[1], "hoist-loads-stores-with-cond-faulting", 37);
  v5 = v30;
  if ( v13 )
  {
    v32 = v10;
    goto LABEL_7;
  }
  v14 = sub_9691B0((const void *)v31.m128i_i64[0], v31.m128i_u64[1], "sink-common-insts", 17);
  v5 = v30;
  if ( v14 )
  {
    v26 = v10;
    goto LABEL_7;
  }
  v15 = sub_9691B0((const void *)v31.m128i_i64[0], v31.m128i_u64[1], "simplify-unreachable", 20);
  v5 = v30;
  if ( v15 )
  {
    v27 = v10;
    goto LABEL_7;
  }
  if ( sub_9691B0((const void *)v31.m128i_i64[0], v31.m128i_u64[1], "speculate-unpredictables", 24) )
  {
    v28 = v10;
    v5 = v30;
    goto LABEL_7;
  }
  if ( !v10 )
  {
    v3 = a1;
    goto LABEL_60;
  }
  if ( !(unsigned __int8)sub_95CB50((const void **)&v43, "bonus-inst-threshold=", 0x15u) )
  {
    v3 = a1;
LABEL_60:
    v21 = sub_C63BB0();
    v48 = 41;
    v22 = v21;
    v24 = v23;
    v47 = "invalid SimplifyCFG pass parameter '{0}' ";
    v49 = v54;
    v50 = 1;
    v51 = 1;
    v52 = &unk_49DB108;
    v53 = &v43;
    v54[0] = &v52;
    sub_23328D0((__int64)v46, (__int64)&v47);
    sub_23058C0((__int64 *)&v44, (__int64)v46, v22, v24);
    v25 = v44;
    *(_BYTE *)(v3 + 24) |= 3u;
    *(_QWORD *)v3 = v25 & 0xFFFFFFFFFFFFFFFELL;
    sub_2240A30(v46);
    return v3;
  }
  v45 = 1;
  v44 = 0;
  if ( !sub_C94210(&v43, 0, &v44) )
  {
    if ( v45 > 0x40 )
    {
      LODWORD(v29) = *(_DWORD *)v44;
      j_j___libc_free_0_0(v44);
    }
    else
    {
      LODWORD(v29) = 0;
      if ( v45 )
        v29 = (__int64)(v44 << (64 - (unsigned __int8)v45)) >> (64 - (unsigned __int8)v45);
    }
    v5 = v41;
    goto LABEL_7;
  }
  v3 = a1;
  v16 = sub_C63BB0();
  v51 = 1;
  v17 = v16;
  v19 = v18;
  v48 = 70;
  v47 = "invalid argument to SimplifyCFG pass bonus-threshold parameter: '{0}' ";
  v49 = v54;
  v50 = 1;
  v52 = &unk_49DB108;
  v53 = &v43;
  v54[0] = &v52;
  sub_23328D0((__int64)v46, (__int64)&v47);
  sub_23058C0(&v42, (__int64)v46, v17, v19);
  v20 = v42;
  *(_BYTE *)(a1 + 24) |= 3u;
  *(_QWORD *)a1 = v20 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v46);
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  return v3;
}
