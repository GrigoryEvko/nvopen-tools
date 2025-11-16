// Function: sub_2B3BE00
// Address: 0x2b3be00
//
void __fastcall sub_2B3BE00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 *v10; // rbx
  __int64 v11; // r15
  __int64 v12; // rdi
  __int64 v13; // rbx
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r14
  int v22; // eax
  unsigned __int64 *v23; // rbx
  __int64 v24; // r14
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  int v27; // [rsp+8h] [rbp-48h]
  unsigned __int64 v28[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( *(unsigned int *)(a1 + 12) < a2 )
  {
    v15 = a2;
    v16 = sub_C8D7D0(a1, a1 + 16, a2, 0x90u, v28, a6);
    v25 = v16;
    do
    {
      while ( 1 )
      {
        if ( v16 )
        {
          *(_DWORD *)(v16 + 8) = 0;
          *(_QWORD *)v16 = v16 + 16;
          *(_DWORD *)(v16 + 12) = 16;
          v19 = *(unsigned int *)(a3 + 8);
          if ( (_DWORD)v19 )
            break;
        }
        v16 += 144;
        if ( !--v15 )
          goto LABEL_22;
      }
      v26 = v16;
      sub_2B0CFB0(v16, a3, v19, v16 + 16, v17, v18);
      v16 = v26 + 144;
      --v15;
    }
    while ( v15 );
LABEL_22:
    v20 = *(unsigned __int64 **)a1;
    v21 = (unsigned __int64 *)(*(_QWORD *)a1 + 144LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v21 )
    {
      do
      {
        v21 -= 18;
        if ( (unsigned __int64 *)*v21 != v21 + 2 )
          _libc_free(*v21);
      }
      while ( v20 != v21 );
      v21 = *(unsigned __int64 **)a1;
    }
    v22 = v28[0];
    if ( (unsigned __int64 *)(a1 + 16) != v21 )
    {
      v27 = v28[0];
      _libc_free((unsigned __int64)v21);
      v22 = v27;
    }
    *(_DWORD *)(a1 + 8) = a2;
    *(_DWORD *)(a1 + 12) = v22;
    *(_QWORD *)a1 = v25;
  }
  else
  {
    v8 = *(unsigned int *)(a1 + 8);
    v9 = a2;
    if ( v8 <= a2 )
      v9 = *(unsigned int *)(a1 + 8);
    if ( v9 )
    {
      v10 = *(unsigned __int64 **)a1;
      v11 = *(_QWORD *)a1 + 144 * v9;
      do
      {
        v12 = (__int64)v10;
        v10 += 18;
        sub_2B0CFB0(v12, a3, v9, a4, a5, a6);
      }
      while ( (unsigned __int64 *)v11 != v10 );
      v8 = *(unsigned int *)(a1 + 8);
    }
    if ( a2 > v8 )
    {
      v13 = *(_QWORD *)a1 + 144 * v8;
      v14 = a2 - v8;
      if ( a2 != v8 )
      {
        do
        {
          if ( v13 )
          {
            *(_DWORD *)(v13 + 8) = 0;
            *(_QWORD *)v13 = v13 + 16;
            *(_DWORD *)(v13 + 12) = 16;
            if ( *(_DWORD *)(a3 + 8) )
              sub_2B0CFB0(v13, a3, v9, a4, a5, a6);
          }
          v13 += 144;
          --v14;
        }
        while ( v14 );
      }
    }
    else if ( a2 < v8 )
    {
      v23 = (unsigned __int64 *)(*(_QWORD *)a1 + 144 * v8);
      v24 = *(_QWORD *)a1 + 144 * a2;
      while ( (unsigned __int64 *)v24 != v23 )
      {
        while ( 1 )
        {
          v23 -= 18;
          if ( (unsigned __int64 *)*v23 == v23 + 2 )
            break;
          _libc_free(*v23);
          if ( (unsigned __int64 *)v24 == v23 )
            goto LABEL_10;
        }
      }
    }
LABEL_10:
    *(_DWORD *)(a1 + 8) = a2;
  }
}
