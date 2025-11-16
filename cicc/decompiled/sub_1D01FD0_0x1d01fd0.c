// Function: sub_1D01FD0
// Address: 0x1d01fd0
//
__int64 __fastcall sub_1D01FD0(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // r11
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int *v7; // r9
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 *v16; // r11
  unsigned int v17; // r12d
  __int64 v19; // rsi
  _QWORD *v20; // rcx
  _QWORD *v21; // rax
  int v22; // r8d
  unsigned int v23; // esi
  unsigned int v24; // edx
  int v25; // esi
  unsigned __int64 v26; // [rsp+8h] [rbp-138h]
  _QWORD *v27; // [rsp+10h] [rbp-130h] BYREF
  __int64 v28; // [rsp+18h] [rbp-128h]
  _QWORD v29[36]; // [rsp+20h] [rbp-120h] BYREF

  v2 = v29;
  v27 = v29;
  v29[0] = a1;
  v29[1] = 0;
  v28 = 0x1000000001LL;
  v5 = 1;
  do
  {
    v6 = *a2;
    v7 = (unsigned int *)&v2[2 * v5 - 2];
    v8 = v7[2];
    v9 = *(unsigned int *)(*(_QWORD *)v7 + 40LL);
    v10 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
    if ( (unsigned int)v8 < (unsigned int)v9 )
    {
      v11 = (unsigned int)(v8 + 1);
      v12 = v11 + (unsigned int)(v9 - v8 - 1) + 1;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v10 + 16 * v8);
        if ( (v13 & 6) == 0 )
        {
          v14 = v13 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !*(_DWORD *)(v6 + 4LL * *(unsigned int *)(v14 + 192)) )
          {
            v7[2] = v11;
            v15 = (unsigned int)v28;
            if ( (unsigned int)v28 >= HIDWORD(v28) )
            {
              v26 = v14;
              sub_16CD150((__int64)&v27, v29, 0, 16, v12, (int)v7);
              v2 = v27;
              v15 = (unsigned int)v28;
              v14 = v26;
            }
            v16 = &v2[2 * v15];
            *v16 = v14;
            v16[1] = 0;
            v5 = (unsigned int)(v28 + 1);
            LODWORD(v28) = v28 + 1;
            goto LABEL_11;
          }
        }
        v8 = v11;
        if ( v12 == v11 + 1 )
          break;
        ++v11;
      }
    }
    v19 = 16 * v9;
    v20 = (_QWORD *)(v10 + v19);
    if ( v10 + v19 == v10 )
    {
      v25 = 1;
      goto LABEL_24;
    }
    v21 = *(_QWORD **)(*(_QWORD *)v7 + 32LL);
    v22 = 0;
    v23 = 0;
    do
    {
      while ( 1 )
      {
        if ( (*v21 & 6) != 0 )
          goto LABEL_18;
        v24 = *(_DWORD *)(v6 + 4LL * *(unsigned int *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 192));
        if ( v24 > v23 )
          break;
        v22 += v24 == v23;
LABEL_18:
        v21 += 2;
        if ( v20 == v21 )
          goto LABEL_22;
      }
      v21 += 2;
      v23 = v24;
      v22 = 0;
    }
    while ( v20 != v21 );
LABEL_22:
    v25 = v22 + v23;
    if ( !v25 )
      v25 = 1;
LABEL_24:
    *(_DWORD *)(v6 + 4LL * *(unsigned int *)(*(_QWORD *)v7 + 192LL)) = v25;
    v5 = (unsigned int)(v28 - 1);
    LODWORD(v28) = v28 - 1;
LABEL_11:
    v2 = v27;
  }
  while ( (_DWORD)v5 );
  v17 = *(_DWORD *)(*a2 + 4LL * *(unsigned int *)(a1 + 192));
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v17;
}
