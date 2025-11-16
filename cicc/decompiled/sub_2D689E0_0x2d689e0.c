// Function: sub_2D689E0
// Address: 0x2d689e0
//
__int64 __fastcall sub_2D689E0(__int64 a1, _QWORD *a2)
{
  int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // esi
  unsigned int v8; // eax
  _QWORD *v9; // rdi
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r15
  _QWORD *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rbx
  _QWORD *v23; // r15
  _QWORD *v24; // r12
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 result; // rax
  unsigned __int64 v29; // r14
  __int64 v30; // r12
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rax
  __int64 v34; // rcx
  unsigned __int64 v35; // rdx
  int v36; // edi
  int v37; // r9d
  _QWORD v38[4]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v39; // [rsp+20h] [rbp-50h] BYREF
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 i; // [rsp+30h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v5 = a2[2];
    v6 = *(_QWORD *)(a1 + 8);
    v7 = v4 - 1;
    v8 = (v4 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v9 = (_QWORD *)(v6 + 32LL * v8);
    v10 = v9[2];
    if ( v5 == v10 )
    {
LABEL_3:
      v39 = 0;
      v40 = 0;
      i = -8192;
      sub_2D57220(v9, -8192);
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v36 = 1;
      while ( v10 != -4096 )
      {
        v37 = v36 + 1;
        v8 = v7 & (v36 + v8);
        v9 = (_QWORD *)(v6 + 32LL * v8);
        v10 = v9[2];
        if ( v5 == v10 )
          goto LABEL_3;
        v36 = v37;
      }
    }
  }
  v11 = *(_QWORD *)(a1 + 32);
  v12 = *(unsigned int *)(a1 + 40);
  v13 = v11 - (_QWORD)a2 + 1064 * v12 - 1064;
  v14 = 0x133F84CFE133F84DLL;
  v15 = 0x133F84CFE133F84DLL * (v13 >> 3);
  if ( v13 > 0 )
  {
    v16 = a2;
    do
    {
      v17 = v16[135];
      v18 = v16[2];
      if ( v17 != v18 )
      {
        LOBYTE(v14) = v18 != 0;
        if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
          sub_BD60C0(v16);
        v16[2] = v17;
        LOBYTE(v13) = v17 != -4096;
        if ( ((v17 != 0) & (unsigned __int8)v13) != 0 && v17 != -8192 )
          sub_BD73F0((__int64)v16);
      }
      v19 = (__int64)(v16 + 136);
      v20 = (__int64)(v16 + 3);
      v16 += 133;
      sub_2D68580(v20, v19, v13, v14);
      --v15;
    }
    while ( v15 );
    LODWORD(v12) = *(_DWORD *)(a1 + 40);
    v11 = *(_QWORD *)(a1 + 32);
  }
  v21 = (unsigned int)(v12 - 1);
  *(_DWORD *)(a1 + 40) = v21;
  v22 = 1064 * v21 + v11;
  v23 = *(_QWORD **)(v22 + 24);
  v24 = &v23[4 * *(unsigned int *)(v22 + 32)];
  if ( v23 != v24 )
  {
    do
    {
      v25 = *(v24 - 2);
      v24 -= 4;
      if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
        sub_BD60C0(v24);
    }
    while ( v23 != v24 );
    v24 = *(_QWORD **)(v22 + 24);
  }
  if ( v24 != (_QWORD *)(v22 + 40) )
    _libc_free((unsigned __int64)v24);
  v26 = *(_QWORD *)(v22 + 16);
  if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
    sub_BD60C0((_QWORD *)v22);
  v27 = *(_QWORD *)(a1 + 32);
  result = v27 + 1064LL * *(unsigned int *)(a1 + 40);
  if ( a2 != (_QWORD *)result )
  {
    v29 = 0x133F84CFE133F84DLL * (((__int64)a2 - v27) >> 3);
    result = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)result )
    {
      v30 = *(unsigned int *)(a1 + 24);
      v31 = *(_QWORD *)(a1 + 8);
      v38[0] = 0;
      v38[1] = 0;
      v38[2] = -4096;
      v32 = v31 + 32 * v30;
      v39 = 0;
      v40 = 0;
      for ( i = -8192; v32 != v31; v31 += 32 )
      {
        v33 = *(_QWORD *)(v31 + 16);
        if ( v33 != -8192 && v33 != -4096 )
          break;
      }
      sub_D68D70(&v39);
      result = sub_D68D70(v38);
      v34 = *(_QWORD *)(a1 + 8) + 32LL * *(unsigned int *)(a1 + 24);
      while ( v34 != v31 )
      {
        v35 = *(unsigned int *)(v31 + 24);
        result = v35;
        if ( v29 < v35 )
        {
          result = (unsigned int)(v35 - 1);
          *(_DWORD *)(v31 + 24) = result;
        }
        for ( v31 += 32; v32 != v31; v31 += 32 )
        {
          result = *(_QWORD *)(v31 + 16);
          if ( result != -4096 && result != -8192 )
            break;
        }
      }
    }
  }
  return result;
}
