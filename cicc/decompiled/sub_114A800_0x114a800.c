// Function: sub_114A800
// Address: 0x114a800
//
unsigned __int64 __fastcall sub_114A800(unsigned __int64 *a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r13
  unsigned __int64 v4; // r8
  __int64 v5; // rax
  __int64 v8; // rcx
  bool v9; // zf
  __int64 v10; // rax
  _BYTE *v11; // rbx
  bool v12; // cf
  unsigned __int64 result; // rax
  _BYTE *v14; // rdx
  unsigned __int64 v15; // rcx
  __int64 v16; // r9
  unsigned __int64 v17; // r14
  _BYTE *v18; // rdx
  unsigned __int64 v19; // rdx
  char v20; // dl
  __int64 v21; // r14
  unsigned __int64 v22; // [rsp+10h] [rbp-40h]
  _BYTE *v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+18h] [rbp-38h]
  unsigned __int64 v25; // [rsp+18h] [rbp-38h]

  v3 = (_BYTE *)a1[1];
  v4 = *a1;
  v5 = (__int64)&v3[-*a1] >> 3;
  if ( v5 == 0xFFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v8 = (__int64)(a1[1] - *a1) >> 3;
  v9 = v5 == 0;
  v10 = 1;
  if ( !v9 )
    v10 = (__int64)(a1[1] - *a1) >> 3;
  v11 = a2;
  v12 = __CFADD__(v8, v10);
  result = v8 + v10;
  v14 = &a2[-v4];
  v15 = v12;
  if ( v12 )
  {
    v21 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !result )
    {
      v16 = 8;
      v17 = 0;
      goto LABEL_7;
    }
    if ( result > 0xFFFFFFFFFFFFFFFLL )
      result = 0xFFFFFFFFFFFFFFFLL;
    v21 = 8 * result;
  }
  v23 = &a2[-v4];
  v25 = *a1;
  result = sub_22077B0(v21);
  v4 = v25;
  v14 = v23;
  v15 = result;
  v17 = result + v21;
  v16 = result + 8;
LABEL_7:
  v18 = &v14[v15];
  if ( v18 )
  {
    *v18 = *(_BYTE *)a3;
    result = *(unsigned int *)(a3 + 4);
    *((_DWORD *)v18 + 1) = result;
  }
  if ( a2 != (_BYTE *)v4 )
  {
    v19 = v15;
    result = v4;
    do
    {
      if ( v19 )
      {
        *(_BYTE *)v19 = *(_BYTE *)result;
        *(_DWORD *)(v19 + 4) = *(_DWORD *)(result + 4);
      }
      result += 8LL;
      v19 += 8LL;
    }
    while ( (_BYTE *)result != a2 );
    v16 = (__int64)&a2[v15 - v4 + 8];
  }
  if ( a2 != v3 )
  {
    result = v16;
    do
    {
      v20 = *v11;
      v11 += 8;
      result += 8LL;
      *(_BYTE *)(result - 8) = v20;
      *(_DWORD *)(result - 4) = *((_DWORD *)v11 - 1);
    }
    while ( v11 != v3 );
    v16 += v3 - a2;
  }
  if ( v4 )
  {
    v22 = v15;
    v24 = v16;
    result = j_j___libc_free_0(v4, a1[2] - v4);
    v15 = v22;
    v16 = v24;
  }
  *a1 = v15;
  a1[1] = v16;
  a1[2] = v17;
  return result;
}
