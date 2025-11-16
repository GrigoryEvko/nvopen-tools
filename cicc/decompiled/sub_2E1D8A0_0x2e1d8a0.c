// Function: sub_2E1D8A0
// Address: 0x2e1d8a0
//
unsigned __int64 __fastcall sub_2E1D8A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v8; // r13
  int v9; // ecx
  unsigned int v10; // eax
  __int64 v11; // r14
  __int64 v12; // rdi
  int v13; // ecx
  int v14; // edx
  _QWORD *v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // r15
  _QWORD *v18; // r14
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 result; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rbx
  __int64 v26; // r13
  unsigned __int64 v27; // r14
  unsigned __int64 v28; // rdx
  __int64 v29; // rsi
  int v30; // ebx
  unsigned int v31; // eax
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *i; // rdx
  _QWORD *v38; // rax
  int v39; // [rsp+Ch] [rbp-34h]
  int v40; // [rsp+Ch] [rbp-34h]

  v7 = *(_QWORD *)(*(_QWORD *)a1 + 104LL) - *(_QWORD *)(*(_QWORD *)a1 + 96LL);
  *(_DWORD *)(a1 + 48) = 0;
  v8 = v7 >> 3;
  *(_DWORD *)(a1 + 104) = v8;
  LOBYTE(v9) = v8;
  v10 = (unsigned int)(v8 + 63) >> 6;
  if ( v10 )
  {
    v11 = v10;
    v12 = 0;
    if ( *(_DWORD *)(a1 + 52) < v10 )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v10, 8u, a5, a6);
      v12 = 8LL * *(unsigned int *)(a1 + 48);
    }
    memset((void *)(*(_QWORD *)(a1 + 40) + v12), 0, 8 * v11);
    *(_DWORD *)(a1 + 48) += (unsigned int)(v8 + 63) >> 6;
    v9 = *(_DWORD *)(a1 + 104);
  }
  v13 = v9 & 0x3F;
  if ( v13 )
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8LL * *(unsigned int *)(a1 + 48) - 8) &= ~(-1LL << v13);
  v14 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v14 || *(_DWORD *)(a1 + 132) )
  {
    v15 = *(_QWORD **)(a1 + 120);
    v16 = 4 * v14;
    v17 = 152LL * *(unsigned int *)(a1 + 136);
    if ( (unsigned int)(4 * v14) < 0x40 )
      v16 = 64;
    v18 = &v15[(unsigned __int64)v17 / 8];
    if ( *(_DWORD *)(a1 + 136) <= v16 )
    {
      for ( ; v15 != v18; v15 += 19 )
      {
        if ( *v15 != -4096 )
        {
          if ( *v15 != -8192 )
          {
            v19 = v15[10];
            if ( (_QWORD *)v19 != v15 + 12 )
              _libc_free(v19);
            v20 = v15[1];
            if ( (_QWORD *)v20 != v15 + 3 )
              _libc_free(v20);
          }
          *v15 = -4096;
        }
      }
      goto LABEL_21;
    }
    while ( 1 )
    {
      while ( *v15 == -8192 )
      {
LABEL_30:
        v15 += 19;
        if ( v18 == v15 )
          goto LABEL_42;
      }
      if ( *v15 != -4096 )
      {
        v23 = v15[10];
        if ( (_QWORD *)v23 != v15 + 12 )
        {
          v39 = v14;
          _libc_free(v23);
          v14 = v39;
        }
        v24 = v15[1];
        if ( (_QWORD *)v24 != v15 + 3 )
        {
          v40 = v14;
          _libc_free(v24);
          v14 = v40;
        }
        goto LABEL_30;
      }
      v15 += 19;
      if ( v18 == v15 )
      {
LABEL_42:
        v29 = *(unsigned int *)(a1 + 136);
        if ( v14 )
        {
          v30 = 64;
          if ( v14 != 1 )
          {
            _BitScanReverse(&v31, v14 - 1);
            v30 = 1 << (33 - (v31 ^ 0x1F));
            if ( v30 < 64 )
              v30 = 64;
          }
          v32 = *(_QWORD **)(a1 + 120);
          if ( (_DWORD)v29 == v30 )
          {
            *(_QWORD *)(a1 + 128) = 0;
            v38 = &v32[19 * v29];
            do
            {
              if ( v32 )
                *v32 = -4096;
              v32 += 19;
            }
            while ( v38 != v32 );
          }
          else
          {
            sub_C7D6A0((__int64)v32, v17, 8);
            v33 = (((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                    | (4 * v30 / 3u + 1)
                    | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                  | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                  | (4 * v30 / 3u + 1)
                  | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
                | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                  | (4 * v30 / 3u + 1)
                  | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                | (4 * v30 / 3u + 1)
                | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1);
            v34 = ((v33 >> 16) | v33) + 1;
            *(_DWORD *)(a1 + 136) = v34;
            v35 = (_QWORD *)sub_C7D670(152 * v34, 8);
            v36 = *(unsigned int *)(a1 + 136);
            *(_QWORD *)(a1 + 128) = 0;
            *(_QWORD *)(a1 + 120) = v35;
            for ( i = &v35[19 * v36]; i != v35; v35 += 19 )
            {
              if ( v35 )
                *v35 = -4096;
            }
          }
          break;
        }
        if ( (_DWORD)v29 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 120), v17, 8);
          *(_QWORD *)(a1 + 120) = 0;
          *(_QWORD *)(a1 + 128) = 0;
          *(_DWORD *)(a1 + 136) = 0;
          break;
        }
LABEL_21:
        *(_QWORD *)(a1 + 128) = 0;
        break;
      }
    }
  }
  result = *(unsigned int *)(a1 + 152);
  v22 = (unsigned int)v8;
  if ( (unsigned int)v8 != result )
  {
    if ( (unsigned int)v8 >= result )
    {
      v25 = *(_QWORD *)(a1 + 160);
      v26 = *(_QWORD *)(a1 + 168);
      v27 = v22 - result;
      if ( v22 > *(unsigned int *)(a1 + 156) )
      {
        sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v22, 0x10u, a5, a6);
        result = *(unsigned int *)(a1 + 152);
      }
      v28 = v27;
      result = *(_QWORD *)(a1 + 144) + 16 * result;
      do
      {
        if ( result )
        {
          *(_QWORD *)result = v25;
          *(_QWORD *)(result + 8) = v26;
        }
        result += 16LL;
        --v28;
      }
      while ( v28 );
      *(_DWORD *)(a1 + 152) += v27;
    }
    else
    {
      *(_DWORD *)(a1 + 152) = v8;
    }
  }
  return result;
}
