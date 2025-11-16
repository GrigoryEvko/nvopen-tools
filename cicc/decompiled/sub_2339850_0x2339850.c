// Function: sub_2339850
// Address: 0x2339850
//
void __fastcall sub_2339850(__int64 a1)
{
  unsigned __int64 v2; // r8
  __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // r14
  unsigned __int64 v8; // rdi
  __int64 v9; // rbx
  __int64 v10; // r14
  unsigned __int64 *v11; // r15
  unsigned __int64 *v12; // r12
  unsigned __int64 v13; // rdi
  __int64 v14; // r14
  unsigned __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // rbx
  _QWORD *v18; // r12
  unsigned __int64 v19; // rdi
  __int64 v20; // r15
  __int64 v21; // [rsp+0h] [rbp-40h]
  __int64 v22; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 120);
  if ( *(_DWORD *)(a1 + 132) )
  {
    v3 = *(unsigned int *)(a1 + 128);
    if ( (_DWORD)v3 )
    {
      v4 = 8 * v3;
      v5 = 0;
      do
      {
        v6 = *(_QWORD **)(v2 + v5);
        if ( v6 != (_QWORD *)-8LL && v6 )
        {
          sub_C7D6A0((__int64)v6, *v6 + 25LL, 8);
          v2 = *(_QWORD *)(a1 + 120);
        }
        v5 += 8;
      }
      while ( v4 != v5 );
    }
  }
  _libc_free(v2);
  if ( *(_DWORD *)(a1 + 108) )
  {
    v7 = *(unsigned int *)(a1 + 104);
    v8 = *(_QWORD *)(a1 + 96);
    if ( (_DWORD)v7 )
    {
      v9 = 0;
      v22 = 8 * v7;
      do
      {
        v10 = *(_QWORD *)(v8 + v9);
        if ( v10 != -8 && v10 )
        {
          v11 = *(unsigned __int64 **)(v10 + 72);
          v12 = &v11[8 * (unsigned __int64)*(unsigned int *)(v10 + 80)];
          v21 = *(_QWORD *)v10 + 153LL;
          if ( v11 != v12 )
          {
            do
            {
              v12 -= 8;
              if ( (unsigned __int64 *)*v12 != v12 + 2 )
                _libc_free(*v12);
            }
            while ( v11 != v12 );
            v12 = *(unsigned __int64 **)(v10 + 72);
          }
          if ( v12 != (unsigned __int64 *)(v10 + 88) )
            _libc_free((unsigned __int64)v12);
          v13 = *(_QWORD *)(v10 + 8);
          if ( v13 != v10 + 24 )
            _libc_free(v13);
          sub_C7D6A0(v10, v21, 8);
          v8 = *(_QWORD *)(a1 + 96);
        }
        v9 += 8;
      }
      while ( v22 != v9 );
    }
  }
  else
  {
    v8 = *(_QWORD *)(a1 + 96);
  }
  _libc_free(v8);
  if ( *(_DWORD *)(a1 + 84) )
  {
    v14 = *(unsigned int *)(a1 + 80);
    v15 = *(_QWORD *)(a1 + 72);
    if ( (_DWORD)v14 )
    {
      v16 = 8 * v14;
      v17 = 0;
      do
      {
        v18 = *(_QWORD **)(v15 + v17);
        if ( v18 && v18 != (_QWORD *)-8LL )
        {
          v19 = v18[1];
          v20 = *v18 + 161LL;
          if ( (_QWORD *)v19 != v18 + 4 )
            _libc_free(v19);
          sub_C7D6A0((__int64)v18, v20, 8);
          v15 = *(_QWORD *)(a1 + 72);
        }
        v17 += 8;
      }
      while ( v16 != v17 );
    }
    _libc_free(v15);
  }
  else
  {
    _libc_free(*(_QWORD *)(a1 + 72));
  }
}
