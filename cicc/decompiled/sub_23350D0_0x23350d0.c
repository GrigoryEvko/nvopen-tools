// Function: sub_23350D0
// Address: 0x23350d0
//
__int64 __fastcall sub_23350D0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  char v4; // bl
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  char v12; // si
  unsigned int v13; // eax
  unsigned int v14; // ebx
  __int64 v15; // rdx
  __int64 v16; // r15
  unsigned __int64 v17; // rax
  char v19; // al
  unsigned int v20; // eax
  unsigned int v21; // ebx
  __int64 v22; // rdx
  __int64 v23; // rax
  char v24; // [rsp+7h] [rbp-D9h]
  int v25; // [rsp+8h] [rbp-D8h]
  __int64 v26; // [rsp+8h] [rbp-D8h]
  __int64 v27; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int64 v28; // [rsp+18h] [rbp-C8h]
  __int64 v29; // [rsp+28h] [rbp-B8h] BYREF
  __m128i v30; // [rsp+30h] [rbp-B0h] BYREF
  int *v31; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v32; // [rsp+48h] [rbp-98h]
  unsigned __int64 v33[4]; // [rsp+50h] [rbp-90h] BYREF
  const char *v34; // [rsp+70h] [rbp-70h] BYREF
  __int64 v35; // [rsp+78h] [rbp-68h]
  _QWORD *v36; // [rsp+80h] [rbp-60h]
  __int64 v37; // [rsp+88h] [rbp-58h]
  char v38; // [rsp+90h] [rbp-50h]
  void *v39; // [rsp+98h] [rbp-48h] BYREF
  __m128i *v40; // [rsp+A0h] [rbp-40h]
  _QWORD v41[7]; // [rsp+A8h] [rbp-38h] BYREF

  v4 = 1;
  v27 = a2;
  v28 = a3;
  v24 = 0;
  v25 = 1;
  if ( !a3 )
  {
LABEL_39:
    v19 = *(_BYTE *)(a1 + 16);
    *(_BYTE *)a1 = v4;
    *(_BYTE *)(a1 + 16) = v19 & 0xFC | 2;
    *(_DWORD *)(a1 + 4) = v25;
    *(_BYTE *)(a1 + 8) = v24;
    return a1;
  }
  while ( 1 )
  {
    v30 = 0u;
    LOBYTE(v34) = 59;
    v5 = sub_C931B0(&v27, &v34, 1u, 0);
    if ( v5 == -1 )
    {
      v7 = v27;
      v5 = v28;
      v8 = 0;
      v9 = 0;
    }
    else
    {
      v6 = v5 + 1;
      v7 = v27;
      if ( v5 + 1 > v28 )
      {
        v6 = v28;
        v8 = 0;
      }
      else
      {
        v8 = v28 - v6;
      }
      v9 = v27 + v6;
      if ( v5 > v28 )
        v5 = v28;
    }
    v30.m128i_i64[0] = v7;
    v30.m128i_i64[1] = v5;
    v27 = v9;
    v28 = v8;
    if ( v5 <= 2 )
      goto LABEL_8;
    if ( *(_WORD *)v7 != 28526 || *(_BYTE *)(v7 + 2) != 45 )
      break;
    v10 = v5 - 3;
    v11 = v7 + 3;
    v30.m128i_i64[0] = v7 + 3;
    v30.m128i_i64[1] = v10;
    if ( v10 != 15 )
    {
      v12 = 0;
LABEL_10:
      if ( v10 != 30
        || *(_QWORD *)v11 ^ 0x6973736572676761LL | *(_QWORD *)(v11 + 8) ^ 0x65726767612D6576LL
        || *(_QWORD *)(v11 + 16) != 0x6C70732D65746167LL
        || *(_DWORD *)(v11 + 24) != 1769239657
        || *(_WORD *)(v11 + 28) != 26478 )
      {
        goto LABEL_11;
      }
      v24 = v12;
      v8 = v28;
      goto LABEL_38;
    }
    if ( *(_QWORD *)(v7 + 3) != 0x662D796669726576LL
      || *(_DWORD *)(v7 + 11) != 1869641833
      || *(_WORD *)(v7 + 15) != 28265
      || *(_BYTE *)(v7 + 17) != 116 )
    {
LABEL_11:
      v13 = sub_C63BB0();
      v35 = 41;
      v14 = v13;
      v16 = v15;
      v34 = "invalid InstCombine pass parameter '{0}' ";
      v36 = v41;
      v37 = 1;
      v38 = 1;
      v39 = &unk_49DB108;
      v40 = &v30;
      v41[0] = &v39;
      sub_23328D0((__int64)v33, (__int64)&v34);
      sub_23058C0((__int64 *)&v31, (__int64)v33, v14, v16);
      v17 = (unsigned __int64)v31;
      *(_BYTE *)(a1 + 16) |= 3u;
      *(_QWORD *)a1 = v17 & 0xFFFFFFFFFFFFFFFELL;
      sub_2240A30(v33);
      return a1;
    }
    v4 = 0;
LABEL_38:
    if ( !v8 )
      goto LABEL_39;
  }
  if ( v5 == 15
    && *(_QWORD *)v7 == 0x662D796669726576LL
    && *(_DWORD *)(v7 + 8) == 1869641833
    && *(_WORD *)(v7 + 12) == 28265
    && *(_BYTE *)(v7 + 14) == 116 )
  {
    v4 = 1;
    goto LABEL_38;
  }
LABEL_8:
  if ( !(unsigned __int8)sub_95CB50((const void **)&v30, "max-iterations=", 0xFu) )
  {
    v10 = v30.m128i_i64[1];
    v11 = v30.m128i_i64[0];
    v12 = 1;
    goto LABEL_10;
  }
  v32 = 1;
  v31 = 0;
  if ( !sub_C94210(&v30, 0, (unsigned __int64 *)&v31) )
  {
    if ( v32 <= 0x40 )
    {
      v25 = (int)v31;
    }
    else
    {
      v25 = *v31;
      j_j___libc_free_0_0((unsigned __int64)v31);
    }
    v8 = v28;
    goto LABEL_38;
  }
  v20 = sub_C63BB0();
  v35 = 69;
  v21 = v20;
  v26 = v22;
  v34 = "invalid argument to InstCombine pass max-iterations parameter: '{0}' ";
  v36 = v41;
  v37 = 1;
  v38 = 1;
  v39 = &unk_49DB108;
  v41[0] = &v39;
  v40 = &v30;
  sub_23328D0((__int64)v33, (__int64)&v34);
  sub_23058C0(&v29, (__int64)v33, v21, v26);
  v23 = v29;
  *(_BYTE *)(a1 + 16) |= 3u;
  *(_QWORD *)a1 = v23 & 0xFFFFFFFFFFFFFFFELL;
  sub_2240A30(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0((unsigned __int64)v31);
  return a1;
}
