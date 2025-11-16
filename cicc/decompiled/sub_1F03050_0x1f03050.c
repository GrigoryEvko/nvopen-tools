// Function: sub_1F03050
// Address: 0x1f03050
//
unsigned __int64 __fastcall sub_1F03050(__int64 a1, unsigned __int64 a2, int a3, _BYTE *a4)
{
  _QWORD *v5; // rax
  __int64 v6; // rsi
  _BYTE *v7; // rsi
  _BYTE *v8; // r9
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // ecx
  unsigned __int64 result; // rax
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rsi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // r9
  int v18; // r9d
  _BYTE *v19; // rsi
  _BYTE *v20; // rdi
  unsigned __int64 v22; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 v23; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v24; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v25; // [rsp+28h] [rbp-48h]
  _BYTE *v26; // [rsp+30h] [rbp-40h]

  v5 = *(_QWORD **)a1;
  v22 = a2;
  v25 = 0;
  v26 = 0;
  v6 = v5[1];
  v24 = 0;
  sub_1F02510((__int64)&v24, 0xF0F0F0F0F0F0F0F1LL * ((v6 - *v5) >> 4));
  v7 = v25;
  if ( v25 == v26 )
  {
    sub_1F027A0((__int64)&v24, v25, &v22);
    v8 = v25;
  }
  else
  {
    if ( v25 )
    {
      *(_QWORD *)v25 = v22;
      v7 = v25;
    }
    v8 = v7 + 8;
    v25 = v7 + 8;
  }
  while ( 1 )
  {
    v9 = *((_QWORD *)v8 - 1);
    v10 = *(_QWORD *)(a1 + 64);
    v25 = v8 - 8;
    v11 = *(_DWORD *)(v9 + 192);
    v22 = v9;
    *(_QWORD *)(v10 + 8LL * (v11 >> 6)) |= 1LL << v11;
    result = v22;
    v13 = *(_QWORD *)(v22 + 112);
    v14 = v13 + 16LL * *(unsigned int *)(v22 + 120);
    if ( v14 != v13 )
      break;
LABEL_16:
    v8 = v25;
    if ( v25 == v24 )
    {
      if ( v25 )
        return j_j___libc_free_0(v25, v26 - v25);
      return result;
    }
  }
  while ( 1 )
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)(a1 + 40);
      v16 = *(_QWORD *)(v14 - 16) & 0xFFFFFFFFFFFFFFF8LL;
      v17 = *(unsigned int *)(v16 + 192);
      result = (*(_QWORD *)(a1 + 48) - v15) >> 2;
      if ( v17 < result )
        break;
LABEL_7:
      v14 -= 16;
      if ( v13 == v14 )
        goto LABEL_16;
    }
    v18 = *(_DWORD *)(v15 + 4 * v17);
    if ( v18 == a3 )
      break;
    result = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8LL * (*(_DWORD *)(v16 + 192) >> 6)) & (1LL << *(_DWORD *)(v16 + 192));
    if ( result || v18 >= a3 )
      goto LABEL_7;
    v23 = *(_QWORD *)(v14 - 16) & 0xFFFFFFFFFFFFFFF8LL;
    v19 = v25;
    if ( v25 == v26 )
    {
      result = (unsigned __int64)sub_1F02EC0((__int64)&v24, v25, &v23);
      goto LABEL_7;
    }
    if ( v25 )
    {
      *(_QWORD *)v25 = v16;
      v19 = v25;
    }
    v14 -= 16;
    v25 = v19 + 8;
    if ( v13 == v14 )
      goto LABEL_16;
  }
  result = (unsigned __int64)a4;
  v20 = v24;
  *a4 = 1;
  if ( v20 )
    return j_j___libc_free_0(v20, v26 - v20);
  return result;
}
