// Function: sub_1BC27C0
// Address: 0x1bc27c0
//
__int64 __fastcall sub_1BC27C0(__int64 a1)
{
  int v2; // eax
  int v3; // esi
  int v4; // ecx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v8; // rdi
  _QWORD *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rsi
  char *v12; // rdx
  char *v13; // rcx
  char *v15; // rsi
  __int64 v16; // rsi
  __int64 v17; // rbx
  unsigned __int64 v18; // rdi
  __int64 v19[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_DWORD *)(a1 + 36);
  v3 = *(_DWORD *)(a1 + 32);
  if ( v2 >= v3 )
  {
    v8 = -1;
    if ( (unsigned __int64)v3 <= 0x124924924924924LL )
      v8 = 112LL * v3 + 8;
    v9 = (_QWORD *)sub_2207820(v8);
    v10 = (__int64)v9;
    if ( v9 )
    {
      *v9 = v3;
      v10 = (__int64)(v9 + 1);
      v11 = v3 - 1LL;
      if ( v11 >= 0 )
      {
        v12 = (char *)(v9 + 1);
        do
        {
          memset(v12, 0, 0x70u);
          v13 = v12 + 48;
          v12 += 112;
          *((_DWORD *)v12 - 17) = 4;
          *((_QWORD *)v12 - 10) = v13;
          *((_DWORD *)v12 - 6) = -1;
          *((_DWORD *)v12 - 5) = -1;
          *((_DWORD *)v12 - 4) = -1;
        }
        while ( v11-- != 0 );
      }
    }
    v19[0] = v10;
    v15 = *(char **)(a1 + 16);
    if ( v15 == *(char **)(a1 + 24) )
    {
      sub_1BC25B0((char **)(a1 + 8), v15, v19);
      v10 = v19[0];
    }
    else
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = v10;
        *(_QWORD *)(a1 + 16) += 8LL;
LABEL_13:
        v5 = 0;
        v4 = 1;
        goto LABEL_3;
      }
      *(_QWORD *)(a1 + 16) = 8;
    }
    if ( v10 )
    {
      v16 = 112LL * *(_QWORD *)(v10 - 8);
      v17 = v10 + v16;
      while ( v17 != v10 )
      {
        v17 -= 112;
        v18 = *(_QWORD *)(v17 + 32);
        if ( v18 != v17 + 48 )
          _libc_free(v18);
      }
      j_j_j___libc_free_0_0(v10 - 8);
    }
    goto LABEL_13;
  }
  v4 = v2 + 1;
  v5 = 112LL * v2;
LABEL_3:
  v6 = *(_QWORD *)(a1 + 16);
  *(_DWORD *)(a1 + 36) = v4;
  return *(_QWORD *)(v6 - 8) + v5;
}
