// Function: sub_2B3B600
// Address: 0x2b3b600
//
__int64 __fastcall sub_2B3B600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // eax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 *v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // r12
  _QWORD *v16; // rax
  __int64 v17; // r8
  unsigned __int64 *v18; // r12
  __int64 v19; // r13
  void *v20; // rdi
  __int64 v21; // r9
  const void *v22; // rsi
  size_t v23; // rdx
  unsigned __int64 *v24; // r13
  int v25; // r13d
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+0h] [rbp-50h]
  int v30; // [rsp+Ch] [rbp-44h]
  int v31; // [rsp+Ch] [rbp-44h]
  unsigned __int64 v32[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *(unsigned int *)(a1 + 8);
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v7 )
  {
    v13 = (unsigned __int64 *)(a1 + 16);
    v14 = sub_C8D7D0(a1, a1 + 16, 0, 0x40u, v32, a6);
    v15 = (unsigned __int64)*(unsigned int *)(a1 + 8) << 6;
    v16 = (_QWORD *)(v15 + v14);
    if ( v15 + v14 )
    {
      *v16 = v16 + 2;
      v16[1] = 0x300000000LL;
      v15 = (unsigned __int64)*(unsigned int *)(a1 + 8) << 6;
    }
    v17 = *(_QWORD *)a1;
    v18 = (unsigned __int64 *)(*(_QWORD *)a1 + v15);
    if ( *(unsigned __int64 **)a1 != v18 )
    {
      v19 = v14;
      do
      {
        if ( v19 )
        {
          v20 = (void *)(v19 + 16);
          *(_DWORD *)(v19 + 8) = 0;
          *(_QWORD *)v19 = v19 + 16;
          *(_DWORD *)(v19 + 12) = 3;
          v21 = *(unsigned int *)(v17 + 8);
          if ( (_DWORD)v21 )
          {
            if ( v19 != v17 )
            {
              v22 = (const void *)(v17 + 16);
              if ( *(_QWORD *)v17 == v17 + 16 )
              {
                v23 = 16LL * (unsigned int)v21;
                if ( (unsigned int)v21 <= 3 )
                  goto LABEL_15;
                v29 = v17;
                v31 = *(_DWORD *)(v17 + 8);
                sub_C8D5F0(v19, (const void *)(v19 + 16), (unsigned int)v21, 0x10u, v17, v21);
                v17 = v29;
                v20 = *(void **)v19;
                LODWORD(v21) = v31;
                v22 = *(const void **)v29;
                v23 = 16LL * *(unsigned int *)(v29 + 8);
                if ( v23 )
                {
LABEL_15:
                  v28 = v17;
                  v30 = v21;
                  memcpy(v20, v22, v23);
                  v17 = v28;
                  LODWORD(v21) = v30;
                }
                *(_DWORD *)(v19 + 8) = v21;
                *(_DWORD *)(v17 + 8) = 0;
              }
              else
              {
                *(_QWORD *)v19 = *(_QWORD *)v17;
                *(_DWORD *)(v19 + 8) = *(_DWORD *)(v17 + 8);
                *(_DWORD *)(v19 + 12) = *(_DWORD *)(v17 + 12);
                *(_QWORD *)v17 = v22;
                *(_DWORD *)(v17 + 12) = 0;
                *(_DWORD *)(v17 + 8) = 0;
              }
            }
          }
        }
        v17 += 64;
        v19 += 64;
      }
      while ( v18 != (unsigned __int64 *)v17 );
      v24 = *(unsigned __int64 **)a1;
      v18 = (unsigned __int64 *)(*(_QWORD *)a1 + ((unsigned __int64)*(unsigned int *)(a1 + 8) << 6));
      if ( *(unsigned __int64 **)a1 != v18 )
      {
        do
        {
          v18 -= 8;
          if ( (unsigned __int64 *)*v18 != v18 + 2 )
            _libc_free(*v18);
        }
        while ( v18 != v24 );
        v18 = *(unsigned __int64 **)a1;
      }
    }
    v25 = v32[0];
    if ( v13 != v18 )
      _libc_free((unsigned __int64)v18);
    v26 = *(_DWORD *)(a1 + 8);
    *(_QWORD *)a1 = v14;
    *(_DWORD *)(a1 + 12) = v25;
    v27 = (unsigned int)(v26 + 1);
    *(_DWORD *)(a1 + 8) = v27;
    return v14 + (v27 << 6) - 64;
  }
  else
  {
    v8 = *(_QWORD *)a1;
    v9 = *(_DWORD *)(a1 + 8);
    v10 = (_QWORD *)(*(_QWORD *)a1 + (v7 << 6));
    if ( v10 )
    {
      *v10 = v10 + 2;
      v10[1] = 0x300000000LL;
      v8 = *(_QWORD *)a1;
      v9 = *(_DWORD *)(a1 + 8);
    }
    v11 = (unsigned int)(v9 + 1);
    *(_DWORD *)(a1 + 8) = v11;
    return v8 + (v11 << 6) - 64;
  }
}
