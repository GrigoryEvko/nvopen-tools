// Function: sub_15C9E30
// Address: 0x15c9e30
//
__int64 __fastcall sub_15C9E30(__int64 a1, _BYTE *a2, size_t a3, _QWORD *a4)
{
  _BYTE *v6; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  const char *v11; // rdi
  __int64 v12; // rdx
  __int64 result; // rax
  unsigned __int8 *v14; // rdi
  size_t v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rsi
  size_t v18; // rdx
  __int64 v19; // rax
  _QWORD v20[2]; // [rsp+0h] [rbp-130h] BYREF
  _QWORD v21[2]; // [rsp+10h] [rbp-120h] BYREF
  char v22; // [rsp+20h] [rbp-110h]
  char v23; // [rsp+21h] [rbp-10Fh]
  _QWORD v24[2]; // [rsp+30h] [rbp-100h] BYREF
  char v25; // [rsp+40h] [rbp-F0h]
  char v26; // [rsp+41h] [rbp-EFh]
  _QWORD v27[2]; // [rsp+50h] [rbp-E0h] BYREF
  char v28; // [rsp+60h] [rbp-D0h]
  char v29; // [rsp+61h] [rbp-CFh]
  _QWORD v30[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+80h] [rbp-B0h]
  __int64 v32; // [rsp+90h] [rbp-A0h]
  __int16 v33; // [rsp+A0h] [rbp-90h]
  char *v34; // [rsp+B0h] [rbp-80h]
  char v35; // [rsp+C0h] [rbp-70h]
  char v36; // [rsp+C1h] [rbp-6Fh]
  __int128 v37; // [rsp+D0h] [rbp-60h]
  __int64 v38; // [rsp+E0h] [rbp-50h]
  unsigned __int8 *v39; // [rsp+F0h] [rbp-40h] BYREF
  size_t n; // [rsp+F8h] [rbp-38h]
  unsigned __int8 src[48]; // [rsp+100h] [rbp-30h] BYREF

  v6 = (_BYTE *)(a1 + 16);
  if ( a2 )
  {
    *(_QWORD *)a1 = v6;
    v8 = (_QWORD *)a3;
    v39 = (unsigned __int8 *)a3;
    if ( a3 > 0xF )
    {
      v19 = sub_22409D0(a1, &v39, 0);
      *(_QWORD *)a1 = v19;
      v6 = (_BYTE *)v19;
      *(_QWORD *)(a1 + 16) = v39;
    }
    else
    {
      if ( a3 == 1 )
      {
        *(_BYTE *)(a1 + 16) = *a2;
LABEL_5:
        *(_QWORD *)(a1 + 8) = v8;
        *((_BYTE *)v8 + (_QWORD)v6) = 0;
        goto LABEL_7;
      }
      if ( !a3 )
        goto LABEL_5;
    }
    memcpy(v6, a2, a3);
    v8 = v39;
    v6 = *(_BYTE **)a1;
    goto LABEL_5;
  }
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
LABEL_7:
  *(_BYTE *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 40) = 0;
  sub_15C9090(a1 + 64, a4);
  if ( !*a4 )
    return sub_2241130(a1 + 32, 0, *(_QWORD *)(a1 + 40), "<UNKNOWN LOCATION>", 18);
  v36 = 1;
  LOWORD(v38) = 265;
  LODWORD(v37) = sub_15C70C0((__int64)a4);
  v34 = ":";
  v35 = 3;
  v33 = 265;
  LODWORD(v32) = sub_15C70B0((__int64)a4);
  v9 = sub_15C70A0((__int64)a4);
  v10 = *(_QWORD *)(v9 - 8LL * *(unsigned int *)(v9 + 8));
  if ( *(_BYTE *)v10 == 15 || (v10 = *(_QWORD *)(v10 - 8LL * *(unsigned int *)(v10 + 8))) != 0 )
  {
    v11 = *(const char **)(v10 - 8LL * *(unsigned int *)(v10 + 8));
    if ( v11 )
      v11 = (const char *)sub_161E970(v11);
    else
      v12 = 0;
  }
  else
  {
    v12 = 0;
    v11 = byte_3F871B3;
  }
  v20[0] = v11;
  v30[0] = v20;
  v20[1] = v12;
  LOWORD(v31) = 773;
  v30[1] = ":";
  v27[1] = v32;
  v27[0] = v30;
  v28 = 2;
  v29 = v33;
  v26 = v35;
  v24[0] = v27;
  v24[1] = v34;
  v25 = 2;
  v21[0] = v24;
  v21[1] = v37;
  v22 = 2;
  v23 = v38;
  sub_16E2FC0(&v39, v21);
  result = (__int64)v39;
  v14 = *(unsigned __int8 **)(a1 + 32);
  if ( v39 == src )
  {
    v18 = n;
    if ( n )
    {
      if ( n == 1 )
      {
        result = src[0];
        *v14 = src[0];
      }
      else
      {
        result = (__int64)memcpy(v14, src, n);
      }
      v18 = n;
      v14 = *(unsigned __int8 **)(a1 + 32);
    }
    *(_QWORD *)(a1 + 40) = v18;
    v14[v18] = 0;
    v14 = v39;
    goto LABEL_16;
  }
  v15 = n;
  v16 = *(_QWORD *)src;
  if ( (unsigned __int8 *)(a1 + 48) == v14 )
  {
    *(_QWORD *)(a1 + 32) = v39;
    *(_QWORD *)(a1 + 40) = v15;
    *(_QWORD *)(a1 + 48) = v16;
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 48);
    *(_QWORD *)(a1 + 32) = v39;
    *(_QWORD *)(a1 + 40) = v15;
    *(_QWORD *)(a1 + 48) = v16;
    if ( v14 )
    {
      v39 = v14;
      *(_QWORD *)src = v17;
      goto LABEL_16;
    }
  }
  v39 = src;
  v14 = src;
LABEL_16:
  n = 0;
  *v14 = 0;
  if ( v39 != src )
    return j_j___libc_free_0(v39, *(_QWORD *)src + 1LL);
  return result;
}
