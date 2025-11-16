// Function: sub_239A530
// Address: 0x239a530
//
__int64 __fastcall sub_239A530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // r14
  unsigned int v11; // r15d
  int v12; // eax
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx
  const void *v16; // rax
  const void *v17; // rsi
  __int64 v18; // rax
  _QWORD *v19; // r13
  __int64 v20; // rcx
  _QWORD *v21; // rcx
  _QWORD *i; // r13
  unsigned __int8 *v23; // rsi
  __int64 v24; // r13
  __int64 v25; // r14
  __int64 v26; // r13
  __int64 v27; // rsi
  int v28; // eax
  int v29; // eax
  _QWORD *v30; // [rsp+8h] [rbp-38h]

  v8 = (void *)(a1 + 16);
  *(_QWORD *)a1 = v8;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  v9 = *(_DWORD *)(a2 + 8);
  if ( v9 && a1 != a2 )
  {
    v16 = *(const void **)a2;
    v17 = (const void *)(a2 + 16);
    if ( v16 == v17 )
    {
      a3 = 40;
      if ( v9 == 1
        || (sub_C8D5F0(a1, v8, v9, 0x28u, v9, a6),
            v8 = *(void **)a1,
            v17 = *(const void **)a2,
            (a3 = 40LL * *(unsigned int *)(a2 + 8)) != 0) )
      {
        memcpy(v8, v17, a3);
      }
      *(_DWORD *)(a1 + 8) = v9;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      *(_QWORD *)a1 = v16;
      v29 = *(_DWORD *)(a2 + 12);
      *(_DWORD *)(a1 + 8) = v9;
      *(_DWORD *)(a1 + 12) = v29;
      *(_QWORD *)a2 = v17;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
  v10 = a1 + 72;
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x100000000LL;
  v11 = *(_DWORD *)(a2 + 64);
  if ( v11 && a1 + 56 != a2 + 56 )
  {
    v18 = *(_QWORD *)(a2 + 56);
    v19 = (_QWORD *)(a2 + 72);
    if ( v18 == a2 + 72 )
    {
      v20 = 1;
      if ( v11 != 1 )
      {
        sub_239A420(a1 + 56, v11, a3, 1, a5, a6);
        v10 = *(_QWORD *)(a1 + 56);
        v19 = *(_QWORD **)(a2 + 56);
        v20 = *(unsigned int *)(a2 + 64);
      }
      v21 = &v19[4 * v20];
      if ( v19 == v21 )
      {
        *(_DWORD *)(a1 + 64) = v11;
      }
      else
      {
        for ( i = v19 + 2; ; i += 4 )
        {
          if ( v10 )
          {
            *(_DWORD *)v10 = *((_DWORD *)i - 4);
            *(_QWORD *)(v10 + 8) = *(i - 1);
            v23 = (unsigned __int8 *)*i;
            *(_QWORD *)(v10 + 16) = *i;
            if ( v23 )
            {
              v30 = v21;
              sub_B976B0((__int64)i, v23, v10 + 16);
              *i = 0;
              v21 = v30;
            }
            *(_QWORD *)(v10 + 24) = i[1];
          }
          v10 += 32;
          if ( v21 == i + 2 )
            break;
        }
        v24 = *(unsigned int *)(a2 + 64);
        v25 = *(_QWORD *)(a2 + 56);
        *(_DWORD *)(a1 + 64) = v11;
        v26 = v25 + 32 * v24;
        while ( v25 != v26 )
        {
          while ( 1 )
          {
            v27 = *(_QWORD *)(v26 - 16);
            v26 -= 32;
            if ( !v27 )
              break;
            sub_B91220(v26 + 16, v27);
            if ( v25 == v26 )
              goto LABEL_25;
          }
        }
      }
LABEL_25:
      *(_DWORD *)(a2 + 64) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 56) = v18;
      v28 = *(_DWORD *)(a2 + 68);
      *(_DWORD *)(a1 + 64) = v11;
      *(_DWORD *)(a1 + 68) = v28;
      *(_QWORD *)(a2 + 56) = v19;
      *(_QWORD *)(a2 + 64) = 0;
    }
  }
  v12 = *(_DWORD *)(a2 + 104);
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  v13 = *(_QWORD *)(a2 + 120);
  *(_DWORD *)(a1 + 104) = v12;
  result = *(unsigned int *)(a2 + 136);
  *(_QWORD *)(a1 + 120) = v13;
  v15 = *(_QWORD *)(a2 + 128);
  ++*(_QWORD *)(a2 + 112);
  *(_QWORD *)(a1 + 112) = 1;
  *(_QWORD *)(a1 + 128) = v15;
  *(_DWORD *)(a1 + 136) = result;
  *(_QWORD *)(a2 + 120) = 0;
  *(_QWORD *)(a2 + 128) = 0;
  *(_DWORD *)(a2 + 136) = 0;
  return result;
}
