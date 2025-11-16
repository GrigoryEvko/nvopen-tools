// Function: sub_354E6C0
// Address: 0x354e6c0
//
__int64 __fastcall sub_354E6C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  unsigned __int64 v9; // rbx
  _QWORD *v10; // rax
  _BYTE *v11; // rdx
  _QWORD *i; // rdx
  __int64 v13; // rax
  int v14; // r15d
  __int64 *v15; // rbx
  __int64 *j; // r13
  _WORD *v17; // rdx
  unsigned __int16 v18; // ax
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int16 *v21; // rdi
  unsigned __int16 *k; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  int v25; // r13d
  unsigned int v26; // eax
  __int64 v27; // rcx
  __int64 v28; // rdi
  _WORD *v29; // rax
  _BYTE *v30; // [rsp+10h] [rbp-70h] BYREF
  __int64 v31; // [rsp+18h] [rbp-68h]
  _BYTE v32[96]; // [rsp+20h] [rbp-60h] BYREF

  if ( *(_BYTE *)(a1 + 40) )
    return sub_354D380((_QWORD *)a1, a2);
  v8 = *(_QWORD *)(a1 + 8);
  v30 = v32;
  v9 = *(unsigned int *)(v8 + 48);
  v31 = 0x600000000LL;
  if ( v9 )
  {
    v10 = v32;
    v11 = v32;
    if ( v9 > 6 )
    {
      sub_C8D5F0((__int64)&v30, v32, v9, 8u, a5, a6);
      v11 = v30;
      v10 = &v30[8 * (unsigned int)v31];
    }
    for ( i = &v11[8 * v9]; i != v10; ++v10 )
    {
      if ( v10 )
        *v10 = 0;
    }
    LODWORD(v31) = v9;
  }
  v13 = *(_QWORD *)(a1 + 32);
  v14 = 0;
  v15 = *(__int64 **)(v13 + 48);
  for ( j = *(__int64 **)(v13 + 56); j != v15; v15 += 32 )
  {
    if ( *(_WORD *)(*v15 + 68) > 0x14u )
    {
      v17 = (_WORD *)v15[2];
      if ( !v17 )
      {
        v28 = *(_QWORD *)(a1 + 32) + 600LL;
        if ( sub_2FF7B70(v28) )
        {
          v29 = sub_2FF7DB0(v28, *v15);
          v15[2] = (__int64)v29;
          v17 = v29;
        }
        else
        {
          v17 = (_WORD *)v15[2];
        }
      }
      v18 = *v17 & 0x1FFF;
      if ( v18 != 0x1FFF )
      {
        v19 = (unsigned __int16)v17[1];
        v14 += v18;
        v20 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
        v21 = (unsigned __int16 *)(v20 + 6 * (v19 + (unsigned __int16)v17[2]));
        for ( k = (unsigned __int16 *)(v20 + 6 * v19); v21 != k; *(_QWORD *)&v30[8 * v23] += *(k - 2) )
        {
          v23 = *k;
          k += 3;
        }
      }
    }
  }
  v24 = *(_QWORD *)(a1 + 8);
  v25 = (*(_DWORD *)(a1 + 484) + v14 - 1) / *(_DWORD *)(a1 + 484);
  v26 = *(_DWORD *)(v24 + 48);
  if ( v26 > 1 )
  {
    v27 = 0;
    do
    {
      if ( v25 < (int)(((unsigned __int64)*(unsigned int *)(*(_QWORD *)(v24 + 32) + 4 * v27 + 40)
                      + *(_QWORD *)&v30[v27 + 8]
                      - 1)
                     / *(unsigned int *)(*(_QWORD *)(v24 + 32) + 4 * v27 + 40)) )
        v25 = ((unsigned __int64)*(unsigned int *)(*(_QWORD *)(v24 + 32) + 4 * v27 + 40) + *(_QWORD *)&v30[v27 + 8] - 1)
            / *(unsigned int *)(*(_QWORD *)(v24 + 32) + 4 * v27 + 40);
      v27 += 8;
    }
    while ( 8LL * (v26 - 1) != v27 );
  }
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return (unsigned int)v25;
}
