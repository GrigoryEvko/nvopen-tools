// Function: sub_2B54CF0
// Address: 0x2b54cf0
//
__int64 __fastcall sub_2B54CF0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  _QWORD *v9; // rdx
  _QWORD *v10; // rcx
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r14
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  int v19; // ebx
  _QWORD *v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+10h] [rbp-60h]
  unsigned __int64 v24; // [rsp+20h] [rbp-50h]
  unsigned __int64 v25; // [rsp+28h] [rbp-48h]
  unsigned __int64 v26[7]; // [rsp+38h] [rbp-38h] BYREF

  v22 = a1 + 16;
  v6 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 0x10u, v26, a6);
  v7 = *(_QWORD *)a1;
  v21 = v6;
  v8 = 16LL * *(unsigned int *)(a1 + 8);
  v25 = *(_QWORD *)a1 + v8;
  if ( v7 != v7 + v8 )
  {
    v9 = (_QWORD *)(v7 + 8);
    v10 = &v6[(unsigned __int64)v8 / 8];
    do
    {
      if ( v6 )
      {
        *v6 = *(v9 - 1);
        v6[1] = *v9;
        *v9 = 0;
      }
      v6 += 2;
      v9 += 2;
    }
    while ( v6 != v10 );
    v24 = *(_QWORD *)a1;
    v25 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v25 )
    {
      do
      {
        v25 -= 16LL;
        v11 = *(_QWORD *)(v25 + 8);
        if ( v11 )
        {
          v12 = *(_QWORD *)(v11 + 144);
          if ( v12 != v11 + 160 )
            _libc_free(v12);
          sub_C7D6A0(*(_QWORD *)(v11 + 120), 8LL * *(unsigned int *)(v11 + 136), 8);
          sub_C7D6A0(*(_QWORD *)(v11 + 88), 16LL * *(unsigned int *)(v11 + 104), 8);
          v13 = *(_QWORD *)(v11 + 8);
          v14 = v13 + 8LL * *(unsigned int *)(v11 + 16);
          if ( v13 != v14 )
          {
            do
            {
              v15 = *(_QWORD *)(v14 - 8);
              v14 -= 8LL;
              if ( v15 )
              {
                v16 = v15 + 160LL * *(_QWORD *)(v15 - 8);
                while ( v15 != v16 )
                {
                  v16 -= 160;
                  v17 = *(_QWORD *)(v16 + 88);
                  if ( v17 != v16 + 104 )
                    _libc_free(v17);
                  v18 = *(_QWORD *)(v16 + 40);
                  if ( v18 != v16 + 56 )
                    _libc_free(v18);
                }
                j_j_j___libc_free_0_0(v15 - 8);
              }
            }
            while ( v13 != v14 );
            v14 = *(_QWORD *)(v11 + 8);
          }
          if ( v14 != v11 + 24 )
            _libc_free(v14);
          j_j___libc_free_0(v11);
        }
      }
      while ( v24 != v25 );
      v25 = *(_QWORD *)a1;
    }
  }
  v19 = v26[0];
  if ( v22 != v25 )
    _libc_free(v25);
  *(_DWORD *)(a1 + 12) = v19;
  *(_QWORD *)a1 = v21;
  return a1;
}
