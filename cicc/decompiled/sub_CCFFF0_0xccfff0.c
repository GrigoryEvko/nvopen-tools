// Function: sub_CCFFF0
// Address: 0xccfff0
//
__int64 __fastcall sub_CCFFF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, int a5, char a6, char a7)
{
  _BYTE *v8; // rcx
  __int64 v9; // rax
  _BYTE *v10; // r15
  size_t v11; // r13
  _QWORD *v12; // rax
  _BYTE *v13; // rdi
  size_t v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // rdi
  __int64 result; // rax
  __int64 v19; // rax
  _QWORD *v20; // rdi
  size_t v21; // rdx
  _BYTE *v22; // [rsp+8h] [rbp-88h]
  size_t v23; // [rsp+18h] [rbp-78h] BYREF
  _QWORD *v24; // [rsp+20h] [rbp-70h] BYREF
  size_t n; // [rsp+28h] [rbp-68h]
  _QWORD src[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v27; // [rsp+40h] [rbp-50h]
  __int64 v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h]

  *(_QWORD *)(a1 + 24) = a4;
  v8 = (_BYTE *)(a1 + 96);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_QWORD *)(a1 + 88) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)a1 = &unk_49DD408;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 32) = a3;
  *(_DWORD *)(a1 + 72) = a5;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 136) = a7;
  v9 = *a2;
  *(_BYTE *)(a1 + 137) = a6;
  *(_QWORD *)(a1 + 8) = v9;
  *(_QWORD *)(a1 + 40) = 0x4100000001LL;
  *(_QWORD *)(a1 + 48) = 0x6200000002LL;
  *(_QWORD *)(a1 + 56) = 0x200000003LL;
  *(_DWORD *)(a1 + 64) = 20;
  v10 = (_BYTE *)a2[29];
  v11 = a2[30];
  v24 = src;
  if ( &v10[v11] && !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v23 = v11;
  if ( v11 > 0xF )
  {
    v19 = sub_22409D0(&v24, &v23, 0);
    v8 = (_BYTE *)(a1 + 96);
    v24 = (_QWORD *)v19;
    v20 = (_QWORD *)v19;
    src[0] = v23;
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(src[0]) = *v10;
      v12 = src;
      goto LABEL_6;
    }
    if ( !v11 )
    {
      v12 = src;
      goto LABEL_6;
    }
    v20 = src;
  }
  v22 = v8;
  memcpy(v20, v10, v11);
  v11 = v23;
  v12 = v24;
  v8 = v22;
LABEL_6:
  n = v11;
  *((_BYTE *)v12 + v11) = 0;
  v13 = *(_BYTE **)(a1 + 80);
  v27 = a2[33];
  v28 = a2[34];
  v29 = a2[35];
  if ( v24 == src )
  {
    v21 = n;
    if ( n )
    {
      if ( n == 1 )
        *v13 = src[0];
      else
        memcpy(v13, src, n);
      v21 = n;
      v13 = *(_BYTE **)(a1 + 80);
    }
    *(_QWORD *)(a1 + 88) = v21;
    v13[v21] = 0;
    v13 = v24;
  }
  else
  {
    v14 = n;
    v15 = src[0];
    if ( v8 == v13 )
    {
      *(_QWORD *)(a1 + 80) = v24;
      *(_QWORD *)(a1 + 88) = v14;
      *(_QWORD *)(a1 + 96) = v15;
    }
    else
    {
      v16 = *(_QWORD *)(a1 + 96);
      *(_QWORD *)(a1 + 80) = v24;
      *(_QWORD *)(a1 + 88) = v14;
      *(_QWORD *)(a1 + 96) = v15;
      if ( v13 )
      {
        v24 = v13;
        src[0] = v16;
        goto LABEL_10;
      }
    }
    v24 = src;
    v13 = src;
  }
LABEL_10:
  n = 0;
  *v13 = 0;
  v17 = v24;
  *(_QWORD *)(a1 + 112) = v27;
  *(_QWORD *)(a1 + 120) = v28;
  result = v29;
  *(_QWORD *)(a1 + 128) = v29;
  if ( v17 != src )
    return j_j___libc_free_0(v17, src[0] + 1LL);
  return result;
}
