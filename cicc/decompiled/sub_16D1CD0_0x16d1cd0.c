// Function: sub_16D1CD0
// Address: 0x16d1cd0
//
__int64 __fastcall sub_16D1CD0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v3; // rbx
  __int64 v4; // r9
  int v5; // eax
  unsigned int v6; // ecx
  unsigned int v7; // r14d
  __int64 v8; // rdx
  void *v10; // rax
  unsigned __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r12
  unsigned __int64 v15; // r15
  __int64 v16; // rdi
  __int64 v17; // r8
  unsigned int v18; // r14d
  unsigned int v19; // r9d
  __int64 v20; // r10
  int v21; // edx
  int v22; // r11d
  unsigned int v23; // eax
  __int64 v24; // rcx
  _QWORD *i; // rsi
  int v26; // ecx
  __int64 v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  unsigned int v31; // [rsp+24h] [rbp-3Ch]
  unsigned __int64 v32; // [rsp+28h] [rbp-38h]
  unsigned __int64 v33; // [rsp+28h] [rbp-38h]

  v2 = a2;
  v3 = a1;
  v4 = *(unsigned int *)(a1 + 8);
  v5 = *(_DWORD *)(a1 + 12);
  v6 = 2 * v4;
  v7 = *(_DWORD *)(a1 + 8);
  v8 = 8 * v4 + 8;
  if ( 4 * v5 > (unsigned int)(3 * v4) )
  {
    v4 = v6;
    v31 = v6;
    v27 = 8LL * v6 + 8;
  }
  else
  {
    if ( (int)v4 - (*(_DWORD *)(a1 + 16) + v5) > (unsigned int)v4 >> 3 )
      return v2;
    v27 = 8 * v4 + 8;
    v31 = *(_DWORD *)(a1 + 8);
  }
  v28 = v8;
  v29 = v4;
  v32 = *(_QWORD *)a1;
  v10 = _libc_calloc(v31 + 1, 0xCu);
  v11 = v32;
  v12 = v29;
  v13 = v28;
  v14 = (__int64)v10;
  if ( v10 )
  {
    v15 = v32;
  }
  else
  {
    if ( v31 == -1 )
      v14 = sub_13A3880(1u);
    else
      sub_16BD1C0("Allocation failed", 1u);
    v15 = *(_QWORD *)a1;
    v7 = *(_DWORD *)(a1 + 8);
    v11 = v32;
    v12 = v29;
    v13 = v28;
  }
  *(_QWORD *)(v14 + 8 * v12) = 2;
  if ( v7 )
  {
    v16 = v7;
    v17 = 0;
    v33 = v11 + v13;
    v18 = a2;
    v19 = v31 - 1;
    do
    {
      v20 = *(_QWORD *)(v15 + 8 * v17);
      if ( v20 && v20 != -8 )
      {
        v21 = 1;
        v22 = *(_DWORD *)(v33 + 4 * v17);
        v23 = v22 & v19;
        v24 = v22 & v19;
        for ( i = (_QWORD *)(v14 + 8 * v24); *i; i = (_QWORD *)(v14 + 8LL * v23) )
        {
          v26 = v21++;
          v23 = v19 & (v26 + v23);
          v24 = v23;
        }
        *i = v20;
        *(_DWORD *)(v14 + v27 + 4 * v24) = v22;
        if ( v2 == (_DWORD)v17 )
          v18 = v23;
      }
      ++v17;
    }
    while ( v16 != v17 );
    v3 = a1;
  }
  else
  {
    v18 = a2;
  }
  v2 = v18;
  _libc_free(v15);
  *(_QWORD *)v3 = v14;
  *(_DWORD *)(v3 + 16) = 0;
  *(_DWORD *)(v3 + 8) = v31;
  return v2;
}
