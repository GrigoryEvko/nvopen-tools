// Function: sub_1E305C0
// Address: 0x1e305c0
//
__int64 __fastcall sub_1E305C0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __int64 v6; // r8
  int v7; // r11d
  __int64 v8; // rcx
  __int64 *v9; // r14
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r9
  __int64 v13; // r13
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdi
  __int64 (*v22)(); // rax
  int v23; // r15d
  __int64 v24; // rax
  __int64 v25; // r15
  int v26; // eax
  int v27; // eax
  int v28; // eax
  __int64 *v29; // rdi
  unsigned int v30; // r13d
  __int64 v31; // rcx
  int v32; // r9d
  __int64 v33; // [rsp+8h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 1792) == a2 )
    return *(_QWORD *)(a1 + 1800);
  v4 = *(unsigned int *)(a1 + 1776);
  v5 = a1 + 1752;
  if ( !(_DWORD)v4 )
  {
    ++*(_QWORD *)(a1 + 1752);
    goto LABEL_9;
  }
  v6 = (unsigned int)(v4 - 1);
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 1760);
  v9 = 0;
  v10 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (__int64 *)(v8 + 16LL * v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    while ( v12 != -8 )
    {
      if ( v12 == -16 && !v9 )
        v9 = v11;
      v10 = v6 & (v7 + v10);
      v11 = (__int64 *)(v8 + 16LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        goto LABEL_4;
      ++v7;
    }
    if ( !v9 )
      v9 = v11;
    v26 = *(_DWORD *)(a1 + 1768);
    ++*(_QWORD *)(a1 + 1752);
    v19 = (unsigned int)(v26 + 1);
    if ( 4 * (int)v19 < (unsigned int)(3 * v4) )
    {
      if ( (int)v4 - *(_DWORD *)(a1 + 1772) - (int)v19 > (unsigned int)v4 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 1768) = v19;
        if ( *v9 != -8 )
          --*(_DWORD *)(a1 + 1772);
        *v9 = a2;
        v20 = 0;
        v9[1] = 0;
        v21 = *(_QWORD *)(a1 + 160);
        v22 = *(__int64 (**)())(*(_QWORD *)v21 + 16LL);
        if ( v22 != sub_16FF750 )
        {
          v4 = a2;
          v20 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _QWORD, __int64))v22)(v21, a2, v19, 0, v6);
        }
        v23 = *(_DWORD *)(a1 + 1784);
        v33 = v20;
        *(_DWORD *)(a1 + 1784) = v23 + 1;
        v24 = sub_22077B0(752);
        v13 = v24;
        if ( v24 )
        {
          v4 = a2;
          sub_1E114F0(v24, a2, *(_QWORD *)(a1 + 160), v33, v23, a1);
        }
        v25 = v9[1];
        v9[1] = v13;
        if ( v25 )
        {
          sub_1E11810(v25, v4);
          j_j___libc_free_0(v25, 752);
        }
        goto LABEL_5;
      }
      sub_1E303B0(v5, v4);
      v27 = *(_DWORD *)(a1 + 1776);
      if ( v27 )
      {
        v28 = v27 - 1;
        v4 = *(_QWORD *)(a1 + 1760);
        v29 = 0;
        v30 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v6 = 1;
        v19 = (unsigned int)(*(_DWORD *)(a1 + 1768) + 1);
        v9 = (__int64 *)(v4 + 16LL * v30);
        v31 = *v9;
        if ( a2 != *v9 )
        {
          while ( v31 != -8 )
          {
            if ( !v29 && v31 == -16 )
              v29 = v9;
            v30 = v28 & (v6 + v30);
            v9 = (__int64 *)(v4 + 16LL * v30);
            v31 = *v9;
            if ( a2 == *v9 )
              goto LABEL_11;
            v6 = (unsigned int)(v6 + 1);
          }
          if ( v29 )
            v9 = v29;
        }
        goto LABEL_11;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 1768);
      BUG();
    }
LABEL_9:
    sub_1E303B0(v5, 2 * v4);
    v15 = *(_DWORD *)(a1 + 1776);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 1760);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = (unsigned int)(*(_DWORD *)(a1 + 1768) + 1);
      v9 = (__int64 *)(v17 + 16LL * v18);
      v4 = *v9;
      if ( a2 != *v9 )
      {
        v32 = 1;
        v6 = 0;
        while ( v4 != -8 )
        {
          if ( !v6 && v4 == -16 )
            v6 = (__int64)v9;
          v18 = v16 & (v32 + v18);
          v9 = (__int64 *)(v17 + 16LL * v18);
          v4 = *v9;
          if ( a2 == *v9 )
            goto LABEL_11;
          ++v32;
        }
        if ( v6 )
          v9 = (__int64 *)v6;
      }
      goto LABEL_11;
    }
    goto LABEL_50;
  }
LABEL_4:
  v13 = v11[1];
LABEL_5:
  *(_QWORD *)(a1 + 1792) = a2;
  *(_QWORD *)(a1 + 1800) = v13;
  return v13;
}
