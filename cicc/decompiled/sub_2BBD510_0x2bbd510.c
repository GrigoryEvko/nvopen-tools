// Function: sub_2BBD510
// Address: 0x2bbd510
//
void __fastcall sub_2BBD510(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  _DWORD *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 v24; // [rsp+8h] [rbp-98h]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+18h] [rbp-88h]
  __int64 v27[3]; // [rsp+28h] [rbp-78h] BYREF
  int v28; // [rsp+40h] [rbp-60h] BYREF
  _DWORD *v29; // [rsp+50h] [rbp-50h] BYREF
  __int64 v30; // [rsp+58h] [rbp-48h]
  _DWORD v31[16]; // [rsp+60h] [rbp-40h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  if ( v5 )
  {
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = v31;
    v24 = 40 * v4;
    v11 = v5 + 40 * v4;
    v12 = 0x400000001LL;
    v13 = v7 + 40 * v10;
    v30 = 0x400000001LL;
    v31[0] = -2;
    if ( v7 != v13 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            v14 = (unsigned int)v30;
            *(_DWORD *)(v7 + 12) = 4;
            if ( (_DWORD)v14 )
              break;
          }
          v7 += 40;
          if ( v13 == v7 )
            goto LABEL_10;
        }
        v25 = v7;
        sub_2B0D430(v7, (__int64)&v29, v14, v12, v8, v9);
        v7 = v25 + 40;
      }
      while ( v13 != v25 + 40 );
LABEL_10:
      if ( v29 != v31 )
        _libc_free((unsigned __int64)v29);
    }
    v29 = v31;
    v15 = v5;
    v27[1] = (__int64)&v28;
    v27[2] = 0x400000001LL;
    v28 = -2;
    v30 = 0x400000001LL;
    for ( v31[0] = -3; v11 != v15; v15 += 40 )
    {
      if ( *(_DWORD *)(v15 + 8) != 1 || (v20 = *(_DWORD **)v15, **(_DWORD **)v15 != v28) && v31[0] != *v20 )
      {
        sub_2B4D8D0(a1, (const void **)v15, v27);
        sub_2B0D510(v27[0], (char **)v15, v16, v17, v18, v19);
        *(_DWORD *)(v27[0] + 32) = *(_DWORD *)(v15 + 32);
        ++*(_DWORD *)(a1 + 16);
        v20 = *(_DWORD **)v15;
      }
      if ( v20 != (_DWORD *)(v15 + 16) )
        _libc_free((unsigned __int64)v20);
    }
    sub_C7D6A0(v5, v24, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29 = v31;
    v30 = 0x400000001LL;
    v22 = v7 + 40 * v21;
    v31[0] = -2;
    if ( v7 != v22 )
    {
      do
      {
        while ( 1 )
        {
          if ( v7 )
          {
            v23 = (unsigned int)v30;
            *(_DWORD *)(v7 + 8) = 0;
            *(_QWORD *)v7 = v7 + 16;
            *(_DWORD *)(v7 + 12) = 4;
            if ( (_DWORD)v23 )
              break;
          }
          v7 += 40;
          if ( v22 == v7 )
            goto LABEL_29;
        }
        v26 = v7;
        sub_2B0D430(v7, (__int64)&v29, v7 + 16, v23, v8, v9);
        v7 = v26 + 40;
      }
      while ( v22 != v26 + 40 );
LABEL_29:
      if ( v29 != v31 )
        _libc_free((unsigned __int64)v29);
    }
  }
}
