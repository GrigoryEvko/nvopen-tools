// Function: sub_2B48820
// Address: 0x2b48820
//
__int64 __fastcall sub_2B48820(__int64 a1)
{
  __int64 v2; // rax
  int v3; // esi
  int v4; // esi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r13
  __int64 v14; // rsi
  _QWORD *v15; // rdx
  _QWORD *v16; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rdi
  char *v20; // rbx
  unsigned __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  int v23; // eax
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  _QWORD *v26; // rbx
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  __int64 v29; // rdi
  char *v30; // rbx
  _QWORD v31[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(int *)(a1 + 76);
  v3 = *(_DWORD *)(a1 + 72);
  if ( (int)v2 >= v3 )
  {
    v9 = -1;
    if ( (unsigned __int64)v3 <= 0xCCCCCCCCCCCCCCLL )
      v9 = 160LL * v3 + 8;
    v10 = (_QWORD *)sub_2207820(v9);
    v13 = v10;
    if ( v10 )
    {
      *v10 = v3;
      v13 = v10 + 1;
      v14 = v3 - 1LL;
      if ( v14 >= 0 )
      {
        v15 = v10 + 1;
        do
        {
          memset(v15, 0, 0xA0u);
          *((_DWORD *)v15 + 13) = 4;
          v15[5] = v15 + 7;
          v16 = v15 + 13;
          v15 += 20;
          *(v15 - 9) = v16;
          *((_DWORD *)v15 - 15) = 4;
          *((_DWORD *)v15 - 4) = -1;
          *((_DWORD *)v15 - 3) = -1;
        }
        while ( v14-- != 0 );
      }
    }
    v18 = *(unsigned int *)(a1 + 16);
    v19 = *(unsigned int *)(a1 + 20);
    v31[0] = v13;
    v20 = (char *)v31;
    v21 = *(_QWORD *)(a1 + 8);
    v22 = v18 + 1;
    v23 = v18;
    if ( v18 + 1 > v19 )
    {
      v29 = a1 + 8;
      if ( v21 > (unsigned __int64)v31 || (unsigned __int64)v31 >= v21 + 8 * v18 )
      {
        sub_2B486F0(v29, v22, v18, v21, v11, v12);
        v18 = *(unsigned int *)(a1 + 16);
        v21 = *(_QWORD *)(a1 + 8);
        v23 = *(_DWORD *)(a1 + 16);
      }
      else
      {
        v30 = (char *)v31 - v21;
        sub_2B486F0(v29, v22, v18, v21, v11, v12);
        v21 = *(_QWORD *)(a1 + 8);
        v18 = *(unsigned int *)(a1 + 16);
        v20 = &v30[v21];
        v23 = *(_DWORD *)(a1 + 16);
      }
    }
    v24 = (_QWORD *)(v21 + 8 * v18);
    if ( v24 )
    {
      *v24 = *(_QWORD *)v20;
      *(_QWORD *)v20 = 0;
      v13 = (_QWORD *)v31[0];
      v23 = *(_DWORD *)(a1 + 16);
    }
    v6 = (unsigned int)(v23 + 1);
    *(_DWORD *)(a1 + 16) = v6;
    if ( v13 )
    {
      v25 = 20LL * *(v13 - 1);
      v26 = &v13[v25];
      while ( v26 != v13 )
      {
        v26 -= 20;
        v27 = v26[11];
        if ( (_QWORD *)v27 != v26 + 13 )
          _libc_free(v27);
        v28 = v26[5];
        if ( (_QWORD *)v28 != v26 + 7 )
          _libc_free(v28);
      }
      j_j_j___libc_free_0_0((unsigned __int64)(v13 - 1));
      v6 = *(unsigned int *)(a1 + 16);
    }
    v5 = 0;
    v4 = 1;
  }
  else
  {
    v4 = v2 + 1;
    v5 = 160 * v2;
    v6 = *(unsigned int *)(a1 + 16);
  }
  v7 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 76) = v4;
  return *(_QWORD *)(v7 + 8 * v6 - 8) + v5;
}
