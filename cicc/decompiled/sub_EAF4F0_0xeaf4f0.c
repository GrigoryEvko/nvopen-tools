// Function: sub_EAF4F0
// Address: 0xeaf4f0
//
__int64 __fastcall sub_EAF4F0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  unsigned int v4; // r15d
  _DWORD *v6; // rax
  size_t v7; // r9
  char *v8; // rdi
  _BYTE *v9; // rax
  _QWORD *v10; // rax
  size_t v11; // rdx
  char *v12; // r8
  __int64 v13; // rsi
  void *v14; // rax
  _DWORD *v15; // rax
  size_t v16; // r9
  char *v17; // r8
  unsigned __int8 *v18; // rcx
  int v19; // eax
  size_t v20; // rsi
  __int64 v21; // rdx
  int v22; // eax
  int v23; // r9d
  __int64 v24; // rcx
  __int64 v25; // rax
  size_t v26; // [rsp+10h] [rbp-120h]
  unsigned __int8 *v27; // [rsp+18h] [rbp-118h]
  __int64 v28; // [rsp+20h] [rbp-110h] BYREF
  __int64 v29; // [rsp+28h] [rbp-108h] BYREF
  _QWORD *v30; // [rsp+30h] [rbp-100h] BYREF
  __int64 v31; // [rsp+38h] [rbp-F8h]
  _QWORD v32[2]; // [rsp+40h] [rbp-F0h] BYREF
  void *src; // [rsp+50h] [rbp-E0h] BYREF
  size_t n; // [rsp+58h] [rbp-D8h]
  _QWORD v35[2]; // [rsp+60h] [rbp-D0h] BYREF
  const char *v36; // [rsp+70h] [rbp-C0h] BYREF
  char v37; // [rsp+90h] [rbp-A0h]
  char v38; // [rsp+91h] [rbp-9Fh]
  _QWORD v39[4]; // [rsp+A0h] [rbp-90h] BYREF
  char v40; // [rsp+C0h] [rbp-70h]
  char v41; // [rsp+C1h] [rbp-6Fh]
  char *v42; // [rsp+D0h] [rbp-60h] BYREF
  size_t v43; // [rsp+D8h] [rbp-58h]
  _QWORD v44[2]; // [rsp+E0h] [rbp-50h] BYREF
  char v45; // [rsp+F0h] [rbp-40h]
  char v46; // [rsp+F1h] [rbp-3Fh]

  v2 = sub_ECD7B0(a1);
  v3 = sub_ECD6A0(v2);
  v30 = v32;
  v31 = 0;
  LOBYTE(v32[0]) = 0;
  src = v35;
  n = 0;
  LOBYTE(v35[0]) = 0;
  v29 = 0;
  v38 = 1;
  v36 = "expected file number";
  v37 = 3;
  if ( (unsigned __int8)sub_ECE130(a1, &v28, &v36) )
    goto LABEL_2;
  v39[0] = "file number less than one";
  v41 = 1;
  v40 = 3;
  if ( (unsigned __int8)sub_ECE070(a1, v28 <= 0, v3, v39) )
    goto LABEL_2;
  v46 = 1;
  v42 = "unexpected token in '.cv_file' directive";
  v45 = 3;
  v6 = (_DWORD *)sub_ECD7B0(a1);
  if ( (unsigned __int8)sub_ECE0A0(a1, *v6 != 3, &v42) )
    goto LABEL_2;
  v4 = sub_EAE3B0((_QWORD *)a1, &v30);
  if ( (_BYTE)v4
    || !(unsigned __int8)sub_ECE2A0(a1, 9)
    && ((v41 = 1,
         v39[0] = "unexpected token in '.cv_file' directive",
         v40 = 3,
         v15 = (_DWORD *)sub_ECD7B0(a1),
         (unsigned __int8)sub_ECE0A0(a1, *v15 != 3, v39))
     || (unsigned __int8)sub_EAE3B0((_QWORD *)a1, &src)
     || (v46 = 1,
         v42 = "expected checksum kind in '.cv_file' directive",
         v45 = 3,
         (unsigned __int8)sub_ECE130(a1, &v29, &v42))
     || (unsigned __int8)sub_ECE000(a1)) )
  {
LABEL_2:
    v4 = 1;
    goto LABEL_3;
  }
  LOBYTE(v44[0]) = 0;
  v7 = n;
  v42 = (char *)v44;
  v43 = 0;
  v8 = (char *)src;
  v27 = (unsigned __int8 *)src;
  if ( !n )
    goto LABEL_13;
  v26 = n;
  sub_22410F0(&v42, (n + 1) >> 1, 0);
  v16 = v26;
  v17 = v42;
  v18 = v27;
  if ( (v26 & 1) == 0 )
    goto LABEL_29;
  v19 = (__int16)word_3F64060[*v27];
  if ( v19 != -1 )
  {
    *v42 = v19;
    v16 = v26 - 1;
    v18 = v27 + 1;
    ++v17;
LABEL_29:
    v20 = v16 >> 1;
    if ( v16 >> 1 )
    {
      v21 = 0;
      do
      {
        v22 = (__int16)word_3F64060[v18[2 * v21]];
        v23 = (__int16)word_3F64060[v18[2 * v21 + 1]];
        if ( v22 == -1 )
          break;
        if ( v23 == -1 )
          break;
        v17[v21++] = v23 | (16 * v22);
      }
      while ( v20 != v21 );
    }
    v17 = v42;
  }
  v8 = (char *)src;
  v7 = v43;
  v9 = src;
  if ( v17 != (char *)v44 )
  {
    if ( src == v35 )
    {
      src = v17;
      n = v43;
      v35[0] = v44[0];
    }
    else
    {
      v24 = v35[0];
      src = v17;
      n = v43;
      v35[0] = v44[0];
      if ( v8 )
      {
        v42 = v8;
        v44[0] = v24;
        goto LABEL_14;
      }
    }
    v42 = (char *)v44;
    v9 = v44;
    goto LABEL_14;
  }
  if ( v43 )
  {
    if ( v43 == 1 )
      *(_BYTE *)src = v44[0];
    else
      memcpy(src, v44, v43);
    v7 = v43;
    v8 = (char *)src;
  }
LABEL_13:
  n = v7;
  v8[v7] = 0;
  v9 = v42;
LABEL_14:
  v43 = 0;
  *v9 = 0;
  if ( v42 != (char *)v44 )
    j_j___libc_free_0(v42, v44[0] + 1LL);
  v10 = *(_QWORD **)(a1 + 224);
  v11 = n;
  v12 = (char *)v10[24];
  v13 = (unsigned int)n;
  v10[34] += (unsigned int)n;
  if ( v10[25] >= (unsigned __int64)&v12[v13] && v12 )
  {
    v10[24] = &v12[v13];
  }
  else
  {
    v25 = sub_9D1E70((__int64)(v10 + 24), v13, v13, 0);
    v11 = n;
    v12 = (char *)v25;
  }
  v14 = memcpy(v12, src, v11);
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD, _QWORD *, __int64, void *, size_t, _QWORD))(**(_QWORD **)(a1 + 232) + 712LL))(
          *(_QWORD *)(a1 + 232),
          (unsigned int)v28,
          v30,
          v31,
          v14,
          n,
          (unsigned __int8)v29) )
  {
    v46 = 1;
    v42 = "file number already allocated";
    v45 = 3;
    v4 = sub_ECDA70(a1, v3, &v42, 0, 0);
  }
LABEL_3:
  if ( src != v35 )
    j_j___libc_free_0(src, v35[0] + 1LL);
  if ( v30 != v32 )
    j_j___libc_free_0(v30, v32[0] + 1LL);
  return v4;
}
