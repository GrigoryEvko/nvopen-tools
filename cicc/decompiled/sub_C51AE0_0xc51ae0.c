// Function: sub_C51AE0
// Address: 0xc51ae0
//
__int64 __fastcall sub_C51AE0(__int64 a1, __int64 a2)
{
  size_t v3; // rdi
  __int64 v5; // r15
  unsigned __int64 v6; // r8
  _BYTE *v7; // rax
  __int64 v8; // r14
  _QWORD *v9; // r9
  unsigned __int64 v10; // rdx
  size_t v11; // rdx
  unsigned __int64 v12; // rcx
  const void **v13; // rax
  size_t v14; // r14
  const void *v15; // r15
  _BYTE *v16; // rsi
  __int64 v17; // rax
  unsigned __int64 v18; // r15
  const void *v19; // rsi
  void *v20; // rdi
  __int64 v21; // r14
  unsigned __int64 v23; // [rsp+0h] [rbp-60h]
  _QWORD *v24; // [rsp+8h] [rbp-58h]
  _BYTE *v25; // [rsp+10h] [rbp-50h] BYREF
  size_t v26; // [rsp+18h] [rbp-48h]
  unsigned __int64 v27; // [rsp+20h] [rbp-40h]
  _BYTE v28[56]; // [rsp+28h] [rbp-38h] BYREF

  v3 = 0;
  v5 = *(_QWORD *)(a2 + 16);
  v6 = *(_QWORD *)(a2 + 8);
  v25 = v28;
  v26 = 0;
  v27 = 8;
  if ( v5 )
  {
    v7 = v28;
    v8 = 0;
    v9 = &v25;
    while ( 1 )
    {
      v7[v3] = 32;
      v11 = v26;
      ++v8;
      v3 = ++v26;
      if ( v5 == v8 )
        break;
      v10 = v11 + 2;
      if ( v10 > v27 )
      {
        v23 = v6;
        v24 = v9;
        sub_C8D290(v9, v28, v10, 1);
        v3 = v26;
        v6 = v23;
        v9 = v24;
      }
      v7 = v25;
    }
    v12 = v27;
  }
  else
  {
    v12 = 8;
  }
  v13 = (const void **)&off_4C5C710;
  if ( v6 <= 1 )
    v13 = (const void **)&off_4C5C720;
  v14 = (size_t)v13[1];
  v15 = *v13;
  if ( v12 < v14 + v3 )
  {
    sub_C8D290(&v25, v28, v14 + v3, 1);
    v3 = v26;
  }
  v16 = v25;
  if ( v14 )
  {
    memcpy(&v25[v3], v15, v14);
    v16 = v25;
    v3 = v26;
  }
  v26 = v14 + v3;
  v17 = sub_CB6200(a1, v16, v14 + v3);
  v18 = *(_QWORD *)(a2 + 8);
  v19 = *(const void **)a2;
  v20 = *(void **)(v17 + 32);
  v21 = v17;
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v20 < v18 )
  {
    sub_CB6200(v17, v19, *(_QWORD *)(a2 + 8));
  }
  else if ( v18 )
  {
    memcpy(v20, v19, *(_QWORD *)(a2 + 8));
    *(_QWORD *)(v21 + 32) += v18;
  }
  if ( v25 != v28 )
    _libc_free(v25, v19);
  return a1;
}
