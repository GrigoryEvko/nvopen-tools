// Function: sub_2E31BE0
// Address: 0x2e31be0
//
__int64 __fastcall sub_2E31BE0(__int64 a1, __int64 a2)
{
  char *v2; // r13
  __int64 v5; // rdi
  void *v6; // rdx
  char *v7; // rdi
  size_t v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rdi
  size_t v12; // rdx
  char *v13; // rsi
  int v15; // eax
  size_t v16; // rdx
  char *v17; // [rsp+0h] [rbp-70h] BYREF
  size_t n; // [rsp+8h] [rbp-68h]
  _QWORD src[2]; // [rsp+10h] [rbp-60h] BYREF
  void *v20[2]; // [rsp+20h] [rbp-50h] BYREF
  char *v21; // [rsp+30h] [rbp-40h]
  __int16 v22; // [rsp+40h] [rbp-30h]

  v2 = (char *)(a1 + 16);
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v5 = *(_QWORD *)(a2 + 32);
  if ( !v5 )
    goto LABEL_8;
  v20[0] = (void *)sub_2E791E0(v5);
  v21 = ":";
  v22 = 773;
  v20[1] = v6;
  sub_CA0F50((__int64 *)&v17, v20);
  v7 = *(char **)a1;
  if ( v17 == (char *)src )
  {
    v16 = n;
    if ( n )
    {
      if ( n == 1 )
        *v7 = src[0];
      else
        memcpy(v7, src, n);
      v16 = n;
      v7 = *(char **)a1;
    }
    *(_QWORD *)(a1 + 8) = v16;
    v7[v16] = 0;
    v7 = v17;
    goto LABEL_6;
  }
  v8 = n;
  v9 = src[0];
  if ( v2 == v7 )
  {
    *(_QWORD *)a1 = v17;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v9;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 16);
    *(_QWORD *)a1 = v17;
    *(_QWORD *)(a1 + 8) = v8;
    *(_QWORD *)(a1 + 16) = v9;
    if ( v7 )
    {
      v17 = v7;
      src[0] = v10;
      goto LABEL_6;
    }
  }
  v17 = (char *)src;
  v7 = (char *)src;
LABEL_6:
  n = 0;
  *v7 = 0;
  if ( v17 != (char *)src )
    j_j___libc_free_0((unsigned __int64)v17);
LABEL_8:
  v11 = *(_QWORD *)(a2 + 16);
  if ( v11 )
  {
    v13 = (char *)sub_BD5D20(v11);
    if ( v12 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490((unsigned __int64 *)a1, v13, v12);
    return a1;
  }
  v15 = *(_DWORD *)(a2 + 24);
  v20[0] = "BB";
  LODWORD(v21) = v15;
  v22 = 2563;
  sub_CA0F50((__int64 *)&v17, v20);
  sub_2241490((unsigned __int64 *)a1, v17, n);
  if ( v17 == (char *)src )
    return a1;
  j_j___libc_free_0((unsigned __int64)v17);
  return a1;
}
