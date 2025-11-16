// Function: sub_2853040
// Address: 0x2853040
//
__int64 __fastcall sub_2853040(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rcx
  unsigned __int64 v9; // r13
  _QWORD *v10; // rcx
  unsigned __int64 v11; // r12
  unsigned __int64 **v12; // rbx
  unsigned __int64 v13; // r15
  unsigned __int64 *v14; // r14
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 *v17; // r14
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  _QWORD *v20; // r15
  _QWORD *v21; // rbx
  __int64 v22; // rax
  int v23; // ebx
  _QWORD *v25; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  _QWORD *v28; // [rsp+18h] [rbp-48h]
  unsigned __int64 v29[7]; // [rsp+28h] [rbp-38h] BYREF

  v26 = a1 + 16;
  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v29, a6);
  v7 = *(_QWORD **)a1;
  v25 = v6;
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(_QWORD *)a1 + v8 * 8;
  if ( v7 != &v7[v8] )
  {
    v10 = &v6[v8];
    do
    {
      if ( v6 )
      {
        *v6 = *v7;
        *v7 = 0;
      }
      ++v6;
      ++v7;
    }
    while ( v6 != v10 );
    v9 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    v28 = *(_QWORD **)a1;
    if ( *(_QWORD *)a1 != v9 )
    {
      do
      {
        v11 = *(_QWORD *)(v9 - 8);
        v9 -= 8LL;
        if ( v11 )
        {
          v12 = *(unsigned __int64 ***)(v11 + 120);
          v13 = (unsigned __int64)&v12[*(unsigned int *)(v11 + 128)];
          if ( v12 != (unsigned __int64 **)v13 )
          {
            do
            {
              v14 = *v12;
              *v12 = 0;
              if ( v14 )
              {
                v15 = v14[8];
                if ( (unsigned __int64 *)v15 != v14 + 10 )
                  _libc_free(v15);
                if ( (unsigned __int64 *)*v14 != v14 + 2 )
                  _libc_free(*v14);
                j_j___libc_free_0((unsigned __int64)v14);
              }
              ++v12;
            }
            while ( (unsigned __int64 **)v13 != v12 );
            v16 = *(_QWORD *)(v11 + 120);
            v13 = v16 + 8LL * *(unsigned int *)(v11 + 128);
            if ( v16 != v13 )
            {
              do
              {
                v17 = *(unsigned __int64 **)(v13 - 8);
                v13 -= 8LL;
                if ( v17 )
                {
                  v18 = v17[8];
                  if ( (unsigned __int64 *)v18 != v17 + 10 )
                    _libc_free(v18);
                  if ( (unsigned __int64 *)*v17 != v17 + 2 )
                    _libc_free(*v17);
                  j_j___libc_free_0((unsigned __int64)v17);
                }
              }
              while ( v16 != v13 );
              v13 = *(_QWORD *)(v11 + 120);
            }
          }
          if ( v13 != v11 + 136 )
            _libc_free(v13);
          v19 = *(_QWORD *)(v11 + 88);
          if ( v19 != v11 + 104 )
            _libc_free(v19);
          v20 = *(_QWORD **)(v11 + 24);
          v21 = &v20[3 * *(unsigned int *)(v11 + 32)];
          if ( v20 != v21 )
          {
            do
            {
              v22 = *(v21 - 1);
              v21 -= 3;
              if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
                sub_BD60C0(v21);
            }
            while ( v20 != v21 );
            v21 = *(_QWORD **)(v11 + 24);
          }
          if ( v21 != (_QWORD *)(v11 + 40) )
            _libc_free((unsigned __int64)v21);
          j_j___libc_free_0(v11);
        }
      }
      while ( v28 != (_QWORD *)v9 );
      v9 = *(_QWORD *)a1;
    }
  }
  v23 = v29[0];
  if ( v26 != v9 )
    _libc_free(v9);
  *(_DWORD *)(a1 + 12) = v23;
  *(_QWORD *)a1 = v25;
  return a1;
}
