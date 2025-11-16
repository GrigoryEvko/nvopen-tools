// Function: sub_3111400
// Address: 0x3111400
//
void __fastcall sub_3111400(unsigned __int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r12
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rbx
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r14
  __int64 v14; // r8
  unsigned __int64 v15; // [rsp+0h] [rbp-40h]
  __int64 v16; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 80);
  if ( *(_DWORD *)(a1 + 92) )
  {
    v3 = *(unsigned int *)(a1 + 88);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v2 + v5);
        if ( v6 != (_QWORD *)-8LL && v6 )
        {
          sub_C7D6A0((__int64)v6, *v6 + 17LL, 8);
          v2 = *(_QWORD *)(a1 + 80);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  v7 = *(unsigned __int64 **)(a1 + 32);
  v8 = &v7[4 * *(unsigned int *)(a1 + 40)];
  if ( v7 != v8 )
  {
    do
    {
      v8 -= 4;
      if ( (unsigned __int64 *)*v8 != v8 + 2 )
        j_j___libc_free_0(*v8);
    }
    while ( v7 != v8 );
    v8 = *(unsigned __int64 **)(a1 + 32);
  }
  if ( v8 != (unsigned __int64 *)(a1 + 48) )
    _libc_free((unsigned __int64)v8);
  v9 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 8);
    v16 = v10 + 72 * v9;
    do
    {
      while ( 1 )
      {
        if ( *(_QWORD *)v10 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v11 = *(_QWORD *)(v10 + 8);
          v12 = v11 + 8LL * *(unsigned int *)(v10 + 16);
          if ( v11 != v12 )
          {
            do
            {
              v13 = *(_QWORD *)(v12 - 8);
              v12 -= 8LL;
              if ( v13 )
              {
                v14 = *(_QWORD *)(v13 + 24);
                if ( v14 )
                {
                  v15 = *(_QWORD *)(v13 + 24);
                  sub_C7D6A0(*(_QWORD *)(v14 + 8), 16LL * *(unsigned int *)(v14 + 24), 8);
                  j_j___libc_free_0(v15);
                }
                j_j___libc_free_0(v13);
              }
            }
            while ( v11 != v12 );
            v12 = *(_QWORD *)(v10 + 8);
          }
          if ( v12 != v10 + 24 )
            break;
        }
        v10 += 72;
        if ( v16 == v10 )
          goto LABEL_28;
      }
      v10 += 72;
      _libc_free(v12);
    }
    while ( v16 != v10 );
LABEL_28:
    v9 = *(unsigned int *)(a1 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 72 * v9, 8);
  j_j___libc_free_0(a1);
}
