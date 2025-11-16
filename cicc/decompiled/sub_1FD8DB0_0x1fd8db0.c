// Function: sub_1FD8DB0
// Address: 0x1fd8db0
//
__int64 __fastcall sub_1FD8DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  bool v7; // cc
  __int64 (*v8)(void); // rax
  unsigned int v9; // r12d
  unsigned int v11; // eax
  unsigned int v12; // esi
  __int64 v13; // rcx
  __int64 v14; // r8
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rdi
  int v18; // r11d
  __int64 *v19; // r10
  int v20; // edi
  int v21; // edi
  unsigned __int8 v22; // [rsp+7h] [rbp-39h]
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  _QWORD v24[5]; // [rsp+18h] [rbp-28h] BYREF

  v7 = *(_BYTE *)(a2 + 16) <= 0x10u;
  v23 = a2;
  if ( v7 )
  {
    v8 = *(__int64 (**)(void))(*(_QWORD *)a1 + 104LL);
    if ( v8 != sub_1FD3460 )
    {
      v22 = a3;
      v11 = v8();
      a3 = v22;
      v9 = v11;
      if ( v11 )
        goto LABEL_6;
      a2 = v23;
    }
  }
  v9 = sub_1FD8980((__int64 *)a1, a2, a3, a4, a5, a6);
  if ( !v9 )
    return 0;
LABEL_6:
  v12 = *(_DWORD *)(a1 + 32);
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_20;
  }
  v13 = v23;
  v14 = *(_QWORD *)(a1 + 16);
  v15 = (v12 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v16 = (__int64 *)(v14 + 16LL * v15);
  v17 = *v16;
  if ( v23 != *v16 )
  {
    v18 = 1;
    v19 = 0;
    while ( v17 != -8 )
    {
      if ( v17 == -16 && !v19 )
        v19 = v16;
      v15 = (v12 - 1) & (v18 + v15);
      v16 = (__int64 *)(v14 + 16LL * v15);
      v17 = *v16;
      if ( v23 == *v16 )
        goto LABEL_8;
      ++v18;
    }
    v20 = *(_DWORD *)(a1 + 24);
    if ( v19 )
      v16 = v19;
    ++*(_QWORD *)(a1 + 8);
    v21 = v20 + 1;
    if ( 4 * v21 < 3 * v12 )
    {
      if ( v12 - *(_DWORD *)(a1 + 28) - v21 > v12 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a1 + 24) = v21;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 28);
        *v16 = v13;
        *((_DWORD *)v16 + 2) = 0;
        goto LABEL_8;
      }
LABEL_21:
      sub_1542080(a1 + 8, v12);
      sub_154CC80(a1 + 8, &v23, v24);
      v16 = (__int64 *)v24[0];
      v13 = v23;
      v21 = *(_DWORD *)(a1 + 24) + 1;
      goto LABEL_16;
    }
LABEL_20:
    v12 *= 2;
    goto LABEL_21;
  }
LABEL_8:
  *((_DWORD *)v16 + 2) = v9;
  *(_QWORD *)(a1 + 144) = sub_1E69D00(*(_QWORD *)(a1 + 56), v9);
  return v9;
}
