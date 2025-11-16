// Function: sub_184D0F0
// Address: 0x184d0f0
//
__int64 __fastcall sub_184D0F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v4; // rax
  unsigned __int8 v5; // dl
  unsigned int v6; // r14d
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 *v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // r13
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rcx
  int v17; // edx
  int v18; // edi
  unsigned int v19; // eax
  __int64 v20; // rsi
  unsigned __int64 v21; // rbx
  int v22; // r8d
  int v23; // r9d
  __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  int v27; // eax
  unsigned __int64 v28[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = (unsigned __int64)sub_1648700(a2);
  v5 = *(_BYTE *)(v4 + 16);
  if ( v5 <= 0x17u )
    goto LABEL_4;
  if ( v5 == 78 )
  {
    v8 = v4;
    v9 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    v10 = v8 | 4;
  }
  else
  {
    if ( v5 != 29 )
      goto LABEL_4;
    v26 = v4;
    v9 = v4 & 0xFFFFFFFFFFFFFFF8LL;
    v10 = v26 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v28[0] = v10;
  if ( !v9 )
    goto LABEL_4;
  v11 = (__int64 *)(v9 - 24);
  v12 = (__int64 *)(v9 - 72);
  if ( (v10 & 4) != 0 )
    v12 = v11;
  v13 = *v12;
  if ( *(_BYTE *)(*v12 + 16) )
    goto LABEL_4;
  if ( sub_15E4F60(*v12) )
    goto LABEL_4;
  sub_15E4B50(v13);
  v6 = v14;
  if ( (_BYTE)v14 )
    goto LABEL_4;
  v15 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(v15 + 8) & 1) != 0 )
  {
    v16 = v15 + 16;
    v17 = 7;
  }
  else
  {
    v16 = *(_QWORD *)(v15 + 16);
    v27 = *(_DWORD *)(v15 + 24);
    if ( !v27 )
      goto LABEL_4;
    v17 = v27 - 1;
  }
  v18 = 1;
  v19 = v17 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
  v20 = *(_QWORD *)(v16 + 8LL * v19);
  if ( v13 != v20 )
  {
    while ( v20 != -8 )
    {
      v19 = v17 & (v18 + v19);
      v20 = *(_QWORD *)(v16 + 8LL * v19);
      if ( v13 == v20 )
        goto LABEL_16;
      ++v18;
    }
    goto LABEL_4;
  }
LABEL_16:
  v21 = 0xAAAAAAAAAAAAAAABLL
      * ((__int64)(a2
                 - ((v28[0] & 0xFFFFFFFFFFFFFFF8LL)
                  - 24LL * (*(_DWORD *)((v28[0] & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF))) >> 3);
  if ( (unsigned int)sub_14DA610(v28) <= (unsigned int)v21
    || (unsigned __int64)(unsigned int)v21 >= *(_QWORD *)(v13 + 96) )
  {
LABEL_4:
    *(_BYTE *)(a1 + 8) = 1;
    return 1;
  }
  if ( (*(_BYTE *)(v13 + 18) & 1) != 0 )
    sub_15E08E0(v13, 0xAAAAAAAAAAAAAAABLL);
  v24 = *(_QWORD *)(v13 + 88) + 40LL * (unsigned int)v21;
  v25 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)v25 >= *(_DWORD *)(a1 + 28) )
  {
    sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 8, v22, v23);
    v25 = *(unsigned int *)(a1 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v25) = v24;
  ++*(_DWORD *)(a1 + 24);
  return v6;
}
