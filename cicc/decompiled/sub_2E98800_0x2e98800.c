// Function: sub_2E98800
// Address: 0x2e98800
//
void __fastcall sub_2E98800(__int64 a1, __int64 a2, int *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v8; // r14d
  unsigned int v9; // r12d
  unsigned int v10; // esi
  __int64 v11; // r9
  __int64 v12; // r14
  int v13; // r12d
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // rsi
  _QWORD *v20; // rdx
  __int64 v21; // rcx
  int v22; // ecx
  unsigned __int64 v23; // rdx
  unsigned __int64 v24; // r8
  int v25; // r14d
  __int64 v26; // rbx
  int v27; // r14d
  unsigned int v29; // [rsp+18h] [rbp-88h]
  unsigned int v30; // [rsp+1Ch] [rbp-84h]
  void *v31; // [rsp+20h] [rbp-80h] BYREF
  __int64 v32; // [rsp+28h] [rbp-78h]
  _BYTE v33[48]; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v34; // [rsp+60h] [rbp-40h]

  v8 = *(_DWORD *)(a1 + 44);
  v31 = v33;
  v9 = (v8 + 63) >> 6;
  v32 = 0x600000000LL;
  if ( v9 > 6 )
  {
    sub_C8D5F0((__int64)&v31, v33, v9, 8u, a5, a6);
    memset(v31, 0, 8LL * v9);
    LODWORD(v32) = (v8 + 63) >> 6;
  }
  else
  {
    if ( v9 && 8LL * v9 )
      memset(v33, 0, 8LL * v9);
    LODWORD(v32) = (v8 + 63) >> 6;
  }
  v10 = *(_DWORD *)(a1 + 16);
  v34 = v8;
  v30 = v10;
  v11 = v10;
  if ( (v10 + 31) >> 5 )
  {
    v12 = 0;
    v29 = v10 - 32 * ((v10 + 31) >> 5);
    do
    {
      v13 = *a3;
      v14 = v12;
      v15 = 0;
      while ( v15 != (_DWORD)v11 )
      {
        if ( v15 + v30 - (_DWORD)v11
          && !_bittest(&v13, v15)
          && (v16 = *(_QWORD *)(a1 + 56) + 2LL * (*(_DWORD *)(*(_QWORD *)(a1 + 8) + v14 + 16) >> 12),
              v17 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + v14 + 16) & 0xFFF,
              v16) )
        {
          do
          {
            v16 += 2;
            *((_QWORD *)v31 + (v17 >> 6)) |= 1LL << v17;
            v17 += *(__int16 *)(v16 - 2);
          }
          while ( *(_WORD *)(v16 - 2) );
          ++v15;
          v14 += 24;
          if ( v15 == 32 )
            break;
        }
        else
        {
          ++v15;
          v14 += 24;
          if ( v15 == 32 )
            break;
        }
      }
      ++a3;
      v12 += 768;
      v11 = (unsigned int)(v11 - 32);
    }
    while ( v29 != (_DWORD)v11 );
    v8 = v34;
  }
  if ( *(_DWORD *)(a2 + 64) < v8 )
  {
    v22 = *(_DWORD *)(a2 + 64) & 0x3F;
    if ( v22 )
      *(_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8) &= ~(-1LL << v22);
    v23 = *(unsigned int *)(a2 + 8);
    v24 = (v8 + 63) >> 6;
    *(_DWORD *)(a2 + 64) = v8;
    if ( v24 != v23 )
    {
      if ( v24 >= v23 )
      {
        v26 = v24 - v23;
        if ( v24 > *(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v24, 8u, v24, v11);
          v23 = *(unsigned int *)(a2 + 8);
        }
        if ( 8 * v26 )
        {
          memset((void *)(*(_QWORD *)a2 + 8 * v23), 0, 8 * v26);
          LODWORD(v23) = *(_DWORD *)(a2 + 8);
        }
        v27 = *(_DWORD *)(a2 + 64);
        *(_DWORD *)(a2 + 8) = v26 + v23;
        v25 = v27 & 0x3F;
        if ( !v25 )
          goto LABEL_17;
LABEL_30:
        *(_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8) &= ~(-1LL << v25);
        goto LABEL_17;
      }
      *(_DWORD *)(a2 + 8) = (v8 + 63) >> 6;
    }
    v25 = v8 & 0x3F;
    if ( !v25 )
      goto LABEL_17;
    goto LABEL_30;
  }
LABEL_17:
  v18 = 0;
  v19 = 8LL * (unsigned int)v32;
  if ( (_DWORD)v32 )
  {
    do
    {
      v20 = (_QWORD *)(v18 + *(_QWORD *)a2);
      v21 = *(_QWORD *)((char *)v31 + v18);
      v18 += 8;
      *v20 |= v21;
    }
    while ( v19 != v18 );
  }
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
}
