// Function: sub_16B0DE0
// Address: 0x16b0de0
//
__int64 __fastcall sub_16B0DE0(__int64 a1, _QWORD *a2, unsigned __int64 *a3)
{
  __int64 v3; // r13
  size_t v7; // rdx
  _BYTE *v8; // rax
  __int64 v9; // rdi
  unsigned __int64 v10; // rbx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rax
  _BYTE *v22; // [rsp+8h] [rbp-38h]

  v3 = a2[1];
  if ( !v3 )
    return 0;
  v7 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v3 >= 0 )
    v7 = a2[1];
  v22 = (_BYTE *)*a2;
  v8 = memchr((const void *)*a2, 61, v7);
  v9 = a1 + 128;
  if ( !v8 || (v10 = v8 - v22, v8 - v22 == -1) )
  {
    v19 = sub_16D1B30(v9, *a2, a2[1]);
    if ( v19 != -1 )
    {
      v20 = *(_QWORD *)(a1 + 128);
      v21 = v20 + 8LL * v19;
      if ( v21 != v20 + 8LL * *(unsigned int *)(a1 + 136) )
        return *(_QWORD *)(*(_QWORD *)v21 + 8LL);
    }
    return 0;
  }
  v11 = v3;
  if ( v3 > v10 )
    v11 = v8 - v22;
  v12 = sub_16D1B30(v9, v22, v11);
  if ( v12 == -1 )
    return 0;
  v13 = *(_QWORD *)(a1 + 128);
  v14 = v13 + 8LL * v12;
  if ( v14 == v13 + 8LL * *(unsigned int *)(a1 + 136) )
    return 0;
  v15 = a2[1];
  v16 = 0;
  if ( v10 + 1 <= v15 )
  {
    v16 = v15 - (v10 + 1);
    v15 = v10 + 1;
  }
  v17 = *a2 + v15;
  a3[1] = v16;
  *a3 = v17;
  if ( a2[1] <= v10 )
    v10 = a2[1];
  a2[1] = v10;
  return *(_QWORD *)(*(_QWORD *)v14 + 8LL);
}
