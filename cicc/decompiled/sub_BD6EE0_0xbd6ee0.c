// Function: sub_BD6EE0
// Address: 0xbd6ee0
//
__int64 __fastcall sub_BD6EE0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // rdi
  int v6; // r14d
  __int64 v7; // r8
  __int64 *v8; // rdx
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  __int64 *v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 result; // rax
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdi
  int v25; // r10d
  __int64 *v26; // r9
  int v27; // eax
  int v28; // eax
  int v29; // r9d
  __int64 *v30; // r8
  __int64 v31; // rdi
  unsigned int v32; // r13d
  __int64 v33; // rsi
  __int64 v34; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+10h] [rbp-30h]

  v2 = sub_BD5C60(a1);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(*(_QWORD *)v2 + 3192LL);
  v5 = *(_QWORD *)v2 + 3168LL;
  if ( !v4 )
  {
    ++*(_QWORD *)(v3 + 3168);
    goto LABEL_36;
  }
  v6 = 1;
  v7 = *(_QWORD *)(v3 + 3176);
  v8 = 0;
  v9 = (v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( *v10 != a1 )
  {
    while ( v11 != -4096 )
    {
      if ( v11 == -8192 && !v8 )
        v8 = v10;
      v9 = (v4 - 1) & (v6 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a1 )
        goto LABEL_3;
      ++v6;
    }
    if ( !v8 )
      v8 = v10;
    v18 = *(_DWORD *)(v3 + 3184);
    ++*(_QWORD *)(v3 + 3168);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(v3 + 3188) - v19 > v4 >> 3 )
      {
LABEL_32:
        *(_DWORD *)(v3 + 3184) = v19;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v3 + 3188);
        *v8 = a1;
        v12 = 0;
        v8[1] = 0;
        goto LABEL_4;
      }
      sub_BD6D00(v5, v4);
      v27 = *(_DWORD *)(v3 + 3192);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = 1;
        v30 = 0;
        v31 = *(_QWORD *)(v3 + 3176);
        v32 = v28 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v19 = *(_DWORD *)(v3 + 3184) + 1;
        v8 = (__int64 *)(v31 + 16LL * v32);
        v33 = *v8;
        if ( *v8 != a1 )
        {
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v30 )
              v30 = v8;
            v32 = v28 & (v29 + v32);
            v8 = (__int64 *)(v31 + 16LL * v32);
            v33 = *v8;
            if ( *v8 == a1 )
              goto LABEL_32;
            ++v29;
          }
          if ( v30 )
            v8 = v30;
        }
        goto LABEL_32;
      }
LABEL_59:
      ++*(_DWORD *)(v3 + 3184);
      BUG();
    }
LABEL_36:
    sub_BD6D00(v5, 2 * v4);
    v20 = *(_DWORD *)(v3 + 3192);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(v3 + 3176);
      v23 = (v20 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v19 = *(_DWORD *)(v3 + 3184) + 1;
      v8 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v8;
      if ( *v8 != a1 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( !v26 && v24 == -8192 )
            v26 = v8;
          v23 = v21 & (v25 + v23);
          v8 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v8;
          if ( *v8 == a1 )
            goto LABEL_32;
          ++v25;
        }
        if ( v26 )
          v8 = v26;
      }
      goto LABEL_32;
    }
    goto LABEL_59;
  }
LABEL_3:
  v12 = (__int64 *)v10[1];
LABEL_4:
  v13 = v12[2];
  v34 = 0;
  v35 = 0;
  v36 = v13;
  if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
    sub_BD6050((unsigned __int64 *)&v34, *v12 & 0xFFFFFFFFFFFFFFF8LL);
  do
  {
    while ( 1 )
    {
      sub_BD60C0(&v34);
      sub_BD6080(&v34, (__int64)v12);
      v15 = (*v12 >> 1) & 3;
      if ( v15 == 1 )
        break;
      if ( (unsigned int)(v15 - 2) <= 1 )
        goto LABEL_8;
LABEL_13:
      v12 = v35;
      if ( !v35 )
        goto LABEL_17;
    }
    v16 = *(__int64 (__fastcall **)(__int64))(*(v12 - 1) + 8);
    if ( v16 == sub_BD61A0 )
    {
LABEL_8:
      v14 = v12[2];
      if ( v14 )
      {
        if ( v14 != -4096 && v14 != -8192 )
          sub_BD60C0(v12);
        v12[2] = 0;
      }
      goto LABEL_13;
    }
    v16((__int64)(v12 - 1));
    v12 = v35;
  }
  while ( v35 );
LABEL_17:
  result = v36;
  if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
    result = sub_BD60C0(&v34);
  if ( (*(_BYTE *)(a1 + 1) & 1) != 0 )
    BUG();
  return result;
}
