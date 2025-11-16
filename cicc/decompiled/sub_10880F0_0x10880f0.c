// Function: sub_10880F0
// Address: 0x10880f0
//
void __fastcall sub_10880F0(__int64 a1)
{
  __int16 *v2; // r11
  char v3; // r13
  __int64 *v4; // r12
  __int64 *v5; // r8
  __int64 v6; // rdi
  _QWORD *v7; // rsi
  __int64 *v8; // rdx
  __int64 v9; // r9
  int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // r14
  _QWORD *v15; // r12
  _QWORD *v16; // r13
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdi
  size_t v21; // r15
  _QWORD *v22; // rax
  __int64 v23; // [rsp-50h] [rbp-50h]
  __int16 v24; // [rsp-3Ah] [rbp-3Ah] BYREF
  __int64 v25; // [rsp-38h] [rbp-38h] BYREF

  if ( !*(_DWORD *)(a1 + 224) )
    return;
  v2 = &v24;
  v3 = 0;
  v4 = *(__int64 **)(a1 + 72);
  v5 = *(__int64 **)(a1 + 80);
  v24 = 256;
  while ( v4 == v5 )
  {
LABEL_17:
    v2 = (__int16 *)((char *)v2 + 1);
    if ( &v25 == (__int64 *)v2 )
      return;
    v3 = *(_BYTE *)v2;
  }
  v6 = *(unsigned int *)(a1 + 232);
  v7 = *(_QWORD **)(a1 + 216);
  v8 = v4;
  v9 = (unsigned int)(v6 - 1);
  while ( 1 )
  {
    v12 = *v8;
    if ( (_DWORD)v6 )
    {
      v10 = v9 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v11 = v7[v10];
      if ( v12 == v11 )
        goto LABEL_6;
      v17 = 1;
      while ( v11 != -4096 )
      {
        v10 = v9 & (v17 + v10);
        v11 = v7[v10];
        if ( v12 == v11 )
          goto LABEL_6;
        ++v17;
      }
    }
    if ( *(_BYTE *)(v12 + 18) != 2 )
      goto LABEL_6;
    v13 = *(_QWORD *)(v12 + 112);
    if ( v13 )
      break;
    if ( *(_DWORD *)(v12 + 12) == -1 )
      goto LABEL_12;
LABEL_6:
    if ( v5 == ++v8 )
      goto LABEL_17;
  }
  if ( !v3 && (*(_BYTE *)(v13 + 37) & 0x10) != 0 )
    goto LABEL_6;
LABEL_12:
  v14 = &v7[v6];
  if ( v7 != v14 )
  {
    while ( 1 )
    {
      v15 = (_QWORD *)*v7;
      v16 = v7;
      if ( *v7 != -8192 && v15 != (_QWORD *)-4096LL )
        break;
      if ( v14 == ++v7 )
        return;
    }
    if ( v7 != v14 )
    {
      do
      {
        v18 = v15[4];
        v19 = (__int64)(v15 + 3);
        if ( (unsigned __int64)(v18 + 1) > v15[5] )
        {
          sub_C8D290((__int64)(v15 + 3), v15 + 6, v18 + 1, 1u, v19, v9);
          v18 = v15[4];
          v19 = (__int64)(v15 + 3);
        }
        *(_BYTE *)(v15[3] + v18) = 46;
        v20 = v15[4] + 1LL;
        v15[4] = v20;
        v21 = *(_QWORD *)(v12 + 32);
        v9 = *(_QWORD *)(v12 + 24);
        if ( v21 + v20 > v15[5] )
        {
          v23 = *(_QWORD *)(v12 + 24);
          sub_C8D290(v19, v15 + 6, v21 + v20, 1u, v19, v9);
          v20 = v15[4];
          v9 = v23;
        }
        if ( v21 )
        {
          memcpy((void *)(v15[3] + v20), (const void *)v9, v21);
          v20 = v15[4];
        }
        v22 = v16 + 1;
        v15[4] = v21 + v20;
        if ( v16 + 1 == v14 )
          break;
        while ( 1 )
        {
          v15 = (_QWORD *)*v22;
          v16 = v22;
          if ( *v22 != -8192 && v15 != (_QWORD *)-4096LL )
            break;
          if ( v14 == ++v22 )
            return;
        }
      }
      while ( v14 != v22 );
    }
  }
}
