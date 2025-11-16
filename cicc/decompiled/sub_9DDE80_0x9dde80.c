// Function: sub_9DDE80
// Address: 0x9dde80
//
__int64 *__fastcall sub_9DDE80(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r10
  __int64 v7; // r13
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r13
  int v11; // r11d
  __int64 v12; // r8
  __int64 *v13; // rdx
  unsigned int v14; // edi
  __int64 *v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // rax
  __int64 v18; // rax
  int v20; // eax
  int v21; // esi
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // r8
  int v25; // eax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rdi
  __int64 *v29; // r8
  unsigned int v30; // r15d
  int v31; // r9d
  __int64 v32; // rsi
  int v33; // r10d
  __int64 *v34; // r9
  __int64 v35[4]; // [rsp+10h] [rbp-60h] BYREF
  char v36; // [rsp+30h] [rbp-40h]
  char v37; // [rsp+31h] [rbp-3Fh]

  v4 = *(_QWORD *)(a2 + 1584);
  if ( *(_QWORD *)(a2 + 1576) == v4 )
  {
    v37 = 1;
    v35[0] = (__int64)"Insufficient function protos";
    v36 = 3;
    sub_9C81F0(a1, a2 + 8, (__int64)v35);
    return a1;
  }
  v5 = *(_QWORD *)(v4 - 8);
  v6 = a2 + 1640;
  *(_QWORD *)(a2 + 1584) = v4 - 8;
  v7 = 8LL * *(_QWORD *)(a2 + 48);
  v8 = *(unsigned int *)(a2 + 64);
  v9 = *(_DWORD *)(a2 + 1664);
  v10 = v7 - v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(a2 + 1640);
    goto LABEL_11;
  }
  v11 = 1;
  v12 = *(_QWORD *)(a2 + 1648);
  v13 = 0;
  v14 = (v9 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v5 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v13 )
        v13 = v15;
      v14 = (v9 - 1) & (v11 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v5 == *v15 )
        goto LABEL_4;
      ++v11;
    }
    if ( !v13 )
      v13 = v15;
    v25 = *(_DWORD *)(a2 + 1656);
    ++*(_QWORD *)(a2 + 1640);
    v16 = (unsigned int)(v25 + 1);
    if ( 4 * (int)v16 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a2 + 1660) - (unsigned int)v16 > v9 >> 3 )
      {
LABEL_13:
        *(_DWORD *)(a2 + 1656) = v16;
        if ( *v13 != -4096 )
          --*(_DWORD *)(a2 + 1660);
        *v13 = v5;
        v17 = v13 + 1;
        v13[1] = 0;
        goto LABEL_5;
      }
      sub_9DDA50(v6, v9);
      v26 = *(_DWORD *)(a2 + 1664);
      if ( v26 )
      {
        v27 = v26 - 1;
        v28 = *(_QWORD *)(a2 + 1648);
        v29 = 0;
        v30 = v27 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v31 = 1;
        v16 = (unsigned int)(*(_DWORD *)(a2 + 1656) + 1);
        v13 = (__int64 *)(v28 + 16LL * v30);
        v32 = *v13;
        if ( v5 != *v13 )
        {
          while ( v32 != -4096 )
          {
            if ( v32 == -8192 && !v29 )
              v29 = v13;
            v30 = v27 & (v31 + v30);
            v13 = (__int64 *)(v28 + 16LL * v30);
            v32 = *v13;
            if ( v5 == *v13 )
              goto LABEL_13;
            ++v31;
          }
          if ( v29 )
            v13 = v29;
        }
        goto LABEL_13;
      }
LABEL_47:
      ++*(_DWORD *)(a2 + 1656);
      BUG();
    }
LABEL_11:
    sub_9DDA50(v6, 2 * v9);
    v20 = *(_DWORD *)(a2 + 1664);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a2 + 1648);
      v23 = (v20 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v16 = (unsigned int)(*(_DWORD *)(a2 + 1656) + 1);
      v13 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v13;
      if ( v5 != *v13 )
      {
        v33 = 1;
        v34 = 0;
        while ( v24 != -4096 )
        {
          if ( !v34 && v24 == -8192 )
            v34 = v13;
          v23 = v21 & (v33 + v23);
          v13 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v13;
          if ( v5 == *v13 )
            goto LABEL_13;
          ++v33;
        }
        if ( v34 )
          v13 = v34;
      }
      goto LABEL_13;
    }
    goto LABEL_47;
  }
LABEL_4:
  v17 = v15 + 1;
LABEL_5:
  *v17 = v10;
  sub_9CE5C0(v35, a2 + 32, (__int64)v13, v16);
  v18 = v35[0] | 1;
  if ( (v35[0] & 0xFFFFFFFFFFFFFFFELL) == 0 )
    v18 = 1;
  *a1 = v18;
  return a1;
}
