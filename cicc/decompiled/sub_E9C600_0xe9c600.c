// Function: sub_E9C600
// Address: 0xe9c600
//
__int64 __fastcall sub_E9C600(__int64 *a1, char a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ecx
  __int64 v7; // rax
  void (*v8)(); // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 i; // rsi
  char v12; // dl
  __int64 v13; // rsi
  unsigned __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  __int64 result; // rax
  _QWORD *v19; // r13
  _QWORD *v20; // rbx
  _QWORD *v21; // rdi
  __int64 v22; // rdi
  __int64 v23; // rdi
  const char *v24; // [rsp+0h] [rbp-80h] BYREF
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v27; // [rsp+18h] [rbp-68h]
  _QWORD *v28; // [rsp+20h] [rbp-60h]
  _QWORD *v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h]
  __int64 v31; // [rsp+38h] [rbp-48h]
  int v32; // [rsp+40h] [rbp-40h]
  __int64 v33; // [rsp+48h] [rbp-38h]
  char v34; // [rsp+50h] [rbp-30h]
  char v35; // [rsp+51h] [rbp-2Fh]
  int v36; // [rsp+54h] [rbp-2Ch]
  __int16 v37; // [rsp+58h] [rbp-28h]

  v6 = *((_DWORD *)a1 + 14);
  if ( v6 && *(_QWORD *)(a1[6] + 16LL * v6 - 8) == *(_QWORD *)(a1[36] + 8) )
  {
    v23 = a1[1];
    LOWORD(v28) = 259;
    v24 = "starting new .cfi frame before finishing the previous one";
    return sub_E66880(v23, a3, (__int64)&v24);
  }
  v24 = 0;
  v37 = 0;
  v7 = *a1;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v36 = 0x7FFFFFFF;
  v35 = a2;
  v8 = *(void (**)())(v7 + 8);
  if ( v8 != nullsub_340 )
  {
    ((void (__fastcall *)(__int64 *, const char **))v8)(a1, &v24);
    v6 = *((_DWORD *)a1 + 14);
  }
  v9 = *(_QWORD *)(a1[1] + 152);
  if ( v9 )
  {
    v10 = *(_QWORD *)(v9 + 360);
    for ( i = *(_QWORD *)(v9 + 368); i != v10; v10 += 104 )
    {
      v12 = *(_BYTE *)(v10 + 32);
      if ( v12 == 7 || (unsigned __int8)(v12 - 4) <= 1u )
        LODWORD(v31) = *(_DWORD *)(v10 + 8);
    }
  }
  v13 = a1[4];
  v14 = *((unsigned int *)a1 + 15);
  v15 = *(_QWORD *)(a1[36] + 8);
  v16 = 0xAAAAAAAAAAAAAAABLL * ((v13 - a1[3]) >> 5);
  v17 = v6;
  if ( v6 >= v14 )
  {
    if ( v14 < (unsigned __int64)v6 + 1 )
    {
      sub_C8D5F0((__int64)(a1 + 6), a1 + 8, v6 + 1LL, 0x10u, v6 + 1LL, a6);
      v17 = *((unsigned int *)a1 + 14);
    }
    result = a1[6] + 16 * v17;
    *(_QWORD *)result = v16;
    *(_QWORD *)(result + 8) = v15;
    v13 = a1[4];
    ++*((_DWORD *)a1 + 14);
  }
  else
  {
    result = a1[6] + 16LL * v6;
    if ( result )
    {
      *(_QWORD *)result = v16;
      *(_QWORD *)(result + 8) = v15;
      v6 = *((_DWORD *)a1 + 14);
      v13 = a1[4];
    }
    *((_DWORD *)a1 + 14) = v6 + 1;
  }
  if ( a1[5] == v13 )
  {
    result = (__int64)sub_E9C240(a1 + 3, (char *)v13, (__int64)&v24);
    v19 = v29;
    v20 = v28;
  }
  else
  {
    if ( v13 )
    {
      *(_QWORD *)v13 = v24;
      *(_QWORD *)(v13 + 8) = v25;
      *(_QWORD *)(v13 + 16) = v26;
      *(_QWORD *)(v13 + 24) = v27;
      *(_QWORD *)(v13 + 32) = v28;
      *(_QWORD *)(v13 + 40) = v29;
      *(_QWORD *)(v13 + 48) = v30;
      *(_QWORD *)(v13 + 56) = v31;
      *(_DWORD *)(v13 + 64) = v32;
      *(_QWORD *)(v13 + 72) = v33;
      *(_BYTE *)(v13 + 80) = v34;
      *(_BYTE *)(v13 + 81) = v35;
      *(_DWORD *)(v13 + 84) = v36;
      *(_BYTE *)(v13 + 88) = v37;
      result = HIBYTE(v37);
      *(_BYTE *)(v13 + 89) = HIBYTE(v37);
      a1[4] += 96;
      return result;
    }
    a1[4] = 96;
    v19 = v29;
    v20 = v28;
  }
  if ( v20 != v19 )
  {
    do
    {
      v21 = (_QWORD *)v20[9];
      result = (__int64)(v20 + 11);
      if ( v21 != v20 + 11 )
        result = j_j___libc_free_0(v21, v20[11] + 1LL);
      v22 = v20[6];
      if ( v22 )
        result = j_j___libc_free_0(v22, v20[8] - v22);
      v20 += 13;
    }
    while ( v19 != v20 );
    v19 = v28;
  }
  if ( v19 )
    return j_j___libc_free_0(v19, v30 - (_QWORD)v19);
  return result;
}
