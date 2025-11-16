// Function: sub_BD7FF0
// Address: 0xbd7ff0
//
__int64 __fastcall sub_BD7FF0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r15d
  __int64 v9; // rcx
  __int64 *v10; // r9
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // r11
  __int64 *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  void (*v18)(); // rax
  __int64 result; // rax
  int v20; // eax
  int v21; // edx
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // eax
  __int64 v26; // rsi
  int v27; // r10d
  __int64 *v28; // r8
  int v29; // eax
  int v30; // eax
  __int64 v31; // rsi
  int v32; // r8d
  __int64 *v33; // rdi
  unsigned int v34; // r14d
  __int64 v35; // rcx
  __int64 v36; // [rsp+0h] [rbp-50h] BYREF
  __int64 *v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+10h] [rbp-40h]

  v4 = sub_BD5C60(a1);
  v5 = *(_QWORD *)v4;
  v6 = *(_DWORD *)(*(_QWORD *)v4 + 3192LL);
  v7 = *(_QWORD *)v4 + 3168LL;
  if ( !v6 )
  {
    ++*(_QWORD *)(v5 + 3168);
    goto LABEL_37;
  }
  v8 = 1;
  v9 = *(_QWORD *)(v5 + 3176);
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( *v12 != a1 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = v12;
      v11 = (v6 - 1) & (v8 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == a1 )
        goto LABEL_3;
      ++v8;
    }
    if ( !v10 )
      v10 = v12;
    v20 = *(_DWORD *)(v5 + 3184);
    ++*(_QWORD *)(v5 + 3168);
    v21 = v20 + 1;
    if ( 4 * (v20 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v5 + 3188) - v21 > v6 >> 3 )
      {
LABEL_33:
        *(_DWORD *)(v5 + 3184) = v21;
        if ( *v10 != -4096 )
          --*(_DWORD *)(v5 + 3188);
        *v10 = a1;
        v14 = 0;
        v10[1] = 0;
        goto LABEL_4;
      }
      sub_BD6D00(v7, v6);
      v29 = *(_DWORD *)(v5 + 3192);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(v5 + 3176);
        v32 = 1;
        v33 = 0;
        v34 = v30 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v21 = *(_DWORD *)(v5 + 3184) + 1;
        v10 = (__int64 *)(v31 + 16LL * v34);
        v35 = *v10;
        if ( *v10 != a1 )
        {
          while ( v35 != -4096 )
          {
            if ( v35 == -8192 && !v33 )
              v33 = v10;
            v34 = v30 & (v32 + v34);
            v10 = (__int64 *)(v31 + 16LL * v34);
            v35 = *v10;
            if ( *v10 == a1 )
              goto LABEL_33;
            ++v32;
          }
          if ( v33 )
            v10 = v33;
        }
        goto LABEL_33;
      }
LABEL_60:
      ++*(_DWORD *)(v5 + 3184);
      BUG();
    }
LABEL_37:
    sub_BD6D00(v7, 2 * v6);
    v22 = *(_DWORD *)(v5 + 3192);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v5 + 3176);
      v25 = (v22 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v21 = *(_DWORD *)(v5 + 3184) + 1;
      v10 = (__int64 *)(v24 + 16LL * v25);
      v26 = *v10;
      if ( *v10 != a1 )
      {
        v27 = 1;
        v28 = 0;
        while ( v26 != -4096 )
        {
          if ( !v28 && v26 == -8192 )
            v28 = v10;
          v25 = v23 & (v27 + v25);
          v10 = (__int64 *)(v24 + 16LL * v25);
          v26 = *v10;
          if ( *v10 == a1 )
            goto LABEL_33;
          ++v27;
        }
        if ( v28 )
          v10 = v28;
      }
      goto LABEL_33;
    }
    goto LABEL_60;
  }
LABEL_3:
  v14 = (__int64 *)v12[1];
LABEL_4:
  v15 = v14[2];
  v36 = 0;
  v37 = 0;
  v38 = v15;
  if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
    sub_BD6050((unsigned __int64 *)&v36, *v14 & 0xFFFFFFFFFFFFFFF8LL);
  do
  {
    while ( 1 )
    {
      sub_BD60C0(&v36);
      sub_BD6080(&v36, (__int64)v14);
      v17 = (*v14 >> 1) & 3;
      if ( v17 != 1 )
      {
        if ( (_DWORD)v17 == 3 )
        {
          v16 = v14[2];
          if ( a2 != v16 )
          {
            if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
              sub_BD60C0(v14);
            v14[2] = a2;
            if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
              sub_BD73F0((__int64)v14);
          }
        }
        goto LABEL_15;
      }
      v18 = *(void (**)())(*(v14 - 1) + 16);
      if ( v18 != nullsub_94 )
        break;
LABEL_15:
      v14 = v37;
      if ( !v37 )
        goto LABEL_19;
    }
    ((void (__fastcall *)(__int64 *, __int64))v18)(v14 - 1, a2);
    v14 = v37;
  }
  while ( v37 );
LABEL_19:
  result = v38;
  if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
    return sub_BD60C0(&v36);
  return result;
}
