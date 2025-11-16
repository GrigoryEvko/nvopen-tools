// Function: sub_2F9A860
// Address: 0x2f9a860
//
__int64 __fastcall sub_2F9A860(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r15
  __int64 v7; // rbx
  __int64 v8; // rdx
  unsigned __int64 v9; // r14
  int v10; // eax
  _QWORD *v11; // r12
  __int64 v12; // rax
  void *v13; // rdi
  unsigned int v14; // r13d
  __int64 result; // rax
  size_t v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // r10
  __int64 v20; // r13
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // r8
  __int64 v24; // r14
  __int64 v25; // rax
  void *v26; // rdi
  unsigned int v27; // ebx
  __int64 v28; // rax
  const void *v29; // rsi
  size_t v30; // rdx
  __int64 v31; // r12
  unsigned __int64 v32; // r13
  unsigned __int64 v33; // rdi
  int v34; // r12d
  __int64 v35; // [rsp+8h] [rbp-68h]
  __int64 v36; // [rsp+8h] [rbp-68h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v38; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+18h] [rbp-58h]
  __int64 v40; // [rsp+20h] [rbp-50h]
  char v41; // [rsp+2Fh] [rbp-41h]
  unsigned __int64 v42[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a2;
  v7 = a1;
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1;
  v10 = *(_DWORD *)(a1 + 8);
  if ( v8 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    if ( v9 <= a2 && a2 < v9 + 56 * v8 )
    {
      v41 = 1;
      v39 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)(a2 - v9) >> 3);
    }
    else
    {
      v39 = -1;
      v41 = 0;
    }
    v40 = a1 + 16;
    v17 = sub_C8D7D0(a1, a1 + 16, v8 + 1, 0x38u, v42, a6);
    v18 = *(unsigned int *)(a1 + 8);
    v9 = v17;
    v19 = *(_QWORD *)a1 + 56 * v18;
    if ( *(_QWORD *)a1 != v19 )
    {
      v20 = v17;
      a6 = a2;
      v21 = *(_QWORD *)a1 + 24LL;
      v22 = v17;
      v23 = a1;
      v24 = *(_QWORD *)a1 + 56 * v18;
      while ( 1 )
      {
        if ( v20 )
        {
          v25 = *(_QWORD *)(v21 - 24);
          v26 = (void *)(v20 + 24);
          *(_DWORD *)(v20 + 16) = 0;
          *(_QWORD *)(v20 + 8) = v20 + 24;
          *(_QWORD *)v20 = v25;
          *(_DWORD *)(v20 + 20) = 2;
          v27 = *(_DWORD *)(v21 - 8);
          if ( v27 )
          {
            if ( v20 + 8 != v21 - 16 )
            {
              v28 = *(_QWORD *)(v21 - 16);
              if ( v28 == v21 )
              {
                v29 = (const void *)v21;
                v30 = 16LL * v27;
                if ( v27 <= 2 )
                  goto LABEL_22;
                v36 = a6;
                v38 = v23;
                sub_C8D5F0(v20 + 8, (const void *)(v20 + 24), v27, 0x10u, v23, a6);
                v26 = *(void **)(v20 + 8);
                v29 = *(const void **)(v21 - 16);
                v23 = v38;
                v30 = 16LL * *(unsigned int *)(v21 - 8);
                a6 = v36;
                if ( v30 )
                {
LABEL_22:
                  v35 = a6;
                  v37 = v23;
                  memcpy(v26, v29, v30);
                  a6 = v35;
                  v23 = v37;
                }
                *(_DWORD *)(v20 + 16) = v27;
                *(_DWORD *)(v21 - 8) = 0;
              }
              else
              {
                *(_QWORD *)(v20 + 8) = v28;
                *(_DWORD *)(v20 + 16) = *(_DWORD *)(v21 - 8);
                *(_DWORD *)(v20 + 20) = *(_DWORD *)(v21 - 4);
                *(_QWORD *)(v21 - 16) = v21;
                *(_DWORD *)(v21 - 4) = 0;
                *(_DWORD *)(v21 - 8) = 0;
              }
            }
          }
        }
        v20 += 56;
        if ( v24 == v21 + 32 )
          break;
        v21 += 56;
      }
      v19 = *(_QWORD *)v23;
      v9 = v22;
      v7 = v23;
      v6 = a6;
      v31 = *(_QWORD *)v23 + 56LL * *(unsigned int *)(v23 + 8);
      if ( v31 != *(_QWORD *)v23 )
      {
        v32 = *(_QWORD *)v23;
        do
        {
          v31 -= 56;
          v33 = *(_QWORD *)(v31 + 8);
          if ( v33 != v31 + 24 )
            _libc_free(v33);
        }
        while ( v31 != v32 );
        v19 = *(_QWORD *)v7;
      }
    }
    v34 = v42[0];
    if ( v19 != v40 )
      _libc_free(v19);
    v8 = *(unsigned int *)(v7 + 8);
    *(_QWORD *)v7 = v9;
    *(_DWORD *)(v7 + 12) = v34;
    v10 = v8;
    if ( v41 )
      v6 = v9 + 56 * v39;
  }
  v11 = (_QWORD *)(v9 + 56 * v8);
  if ( v11 )
  {
    v12 = *(_QWORD *)v6;
    v13 = v11 + 3;
    v11[1] = v11 + 3;
    *v11 = v12;
    v11[2] = 0x200000000LL;
    v14 = *(_DWORD *)(v6 + 16);
    if ( v14 && v11 + 1 != (_QWORD *)(v6 + 8) )
    {
      v16 = 16LL * v14;
      if ( v14 <= 2
        || (sub_C8D5F0((__int64)(v11 + 1), v11 + 3, v14, 0x10u, (__int64)(v11 + 1), a6),
            v13 = (void *)v11[1],
            (v16 = 16LL * *(unsigned int *)(v6 + 16)) != 0) )
      {
        memcpy(v13, *(const void **)(v6 + 8), v16);
      }
      *((_DWORD *)v11 + 4) = v14;
    }
    v10 = *(_DWORD *)(v7 + 8);
  }
  result = (unsigned int)(v10 + 1);
  *(_DWORD *)(v7 + 8) = result;
  return result;
}
