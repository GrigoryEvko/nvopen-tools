// Function: sub_2D503D0
// Address: 0x2d503d0
//
__int64 __fastcall sub_2D503D0(__int64 a1)
{
  int v2; // ecx
  unsigned __int64 v3; // r8
  __int64 v4; // r12
  __int64 v5; // r12
  __int64 v6; // rbx
  _QWORD *v7; // rdi
  __int64 v8; // r14
  unsigned __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // r14
  unsigned __int64 *v12; // r15
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // rdi
  __int64 v15; // r14
  unsigned __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rbx
  _QWORD *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 v21; // r15
  __int64 v23; // [rsp+0h] [rbp-40h]
  __int64 v24; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 308);
  v3 = *(_QWORD *)(a1 + 296);
  *(_QWORD *)a1 = &unk_4A26438;
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 304);
    if ( (_DWORD)v4 )
    {
      v5 = 8 * v4;
      v6 = 0;
      do
      {
        v7 = *(_QWORD **)(v3 + v6);
        if ( v7 != (_QWORD *)-8LL && v7 )
        {
          sub_C7D6A0((__int64)v7, *v7 + 25LL, 8);
          v3 = *(_QWORD *)(a1 + 296);
        }
        v6 += 8;
      }
      while ( v5 != v6 );
    }
  }
  _libc_free(v3);
  if ( *(_DWORD *)(a1 + 284) )
  {
    v8 = *(unsigned int *)(a1 + 280);
    v9 = *(_QWORD *)(a1 + 272);
    if ( (_DWORD)v8 )
    {
      v10 = 0;
      v24 = 8 * v8;
      do
      {
        v11 = *(_QWORD *)(v9 + v10);
        if ( v11 != -8 && v11 )
        {
          v12 = *(unsigned __int64 **)(v11 + 72);
          v13 = &v12[8 * (unsigned __int64)*(unsigned int *)(v11 + 80)];
          v23 = *(_QWORD *)v11 + 153LL;
          if ( v12 != v13 )
          {
            do
            {
              v13 -= 8;
              if ( (unsigned __int64 *)*v13 != v13 + 2 )
                _libc_free(*v13);
            }
            while ( v12 != v13 );
            v13 = *(unsigned __int64 **)(v11 + 72);
          }
          if ( v13 != (unsigned __int64 *)(v11 + 88) )
            _libc_free((unsigned __int64)v13);
          v14 = *(_QWORD *)(v11 + 8);
          if ( v14 != v11 + 24 )
            _libc_free(v14);
          sub_C7D6A0(v11, v23, 8);
          v9 = *(_QWORD *)(a1 + 272);
        }
        v10 += 8;
      }
      while ( v24 != v10 );
    }
  }
  else
  {
    v9 = *(_QWORD *)(a1 + 272);
  }
  _libc_free(v9);
  if ( *(_DWORD *)(a1 + 260) )
  {
    v15 = *(unsigned int *)(a1 + 256);
    v16 = *(_QWORD *)(a1 + 248);
    if ( (_DWORD)v15 )
    {
      v17 = 8 * v15;
      v18 = 0;
      do
      {
        v19 = *(_QWORD **)(v16 + v18);
        if ( v19 && v19 != (_QWORD *)-8LL )
        {
          v20 = v19[1];
          v21 = *v19 + 161LL;
          if ( (_QWORD *)v20 != v19 + 4 )
            _libc_free(v20);
          sub_C7D6A0((__int64)v19, v21, 8);
          v16 = *(_QWORD *)(a1 + 248);
        }
        v18 += 8;
      }
      while ( v17 != v18 );
    }
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 248);
  }
  _libc_free(v16);
  return sub_BB9280(a1);
}
