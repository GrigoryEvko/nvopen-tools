// Function: sub_2BFE2A0
// Address: 0x2bfe2a0
//
__int64 __fastcall sub_2BFE2A0(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax
  bool v7; // cc
  _QWORD *v8; // rax
  __int64 result; // rax
  unsigned int v10; // esi
  __int64 v11; // rbx
  __int64 v12; // r9
  int v13; // r14d
  _QWORD *v14; // rdi
  unsigned int v15; // r8d
  _QWORD *v16; // rdx
  __int64 v17; // rcx
  __int64 *v18; // rdx
  int v19; // edx
  int v20; // r8d
  __int64 v21; // r9
  unsigned int v22; // esi
  int v23; // ecx
  __int64 v24; // rdx
  int v25; // ecx
  int v26; // edx
  int v27; // esi
  __int64 v28; // r8
  _QWORD *v29; // r9
  unsigned int v30; // r13d
  int v31; // r10d
  __int64 v32; // rdx
  int v33; // r11d
  _QWORD *v34; // r10
  __int64 v35; // [rsp+8h] [rbp-28h]
  __int64 v36; // [rsp+8h] [rbp-28h]

  v4 = *(_DWORD *)(a2 + 160);
  if ( v4 - 13 <= 0x11 )
  {
    result = sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
    v10 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    if ( v10 )
    {
      v12 = *(_QWORD *)(a1 + 8);
      v13 = 1;
      v14 = 0;
      v15 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = (_QWORD *)(v12 + 16LL * v15);
      v17 = *v16;
      if ( v11 == *v16 )
      {
LABEL_15:
        v18 = v16 + 1;
LABEL_16:
        *v18 = result;
        return result;
      }
      while ( v17 != -4096 )
      {
        if ( !v14 && v17 == -8192 )
          v14 = v16;
        v15 = (v10 - 1) & (v13 + v15);
        v16 = (_QWORD *)(v12 + 16LL * v15);
        v17 = *v16;
        if ( v11 == *v16 )
          goto LABEL_15;
        ++v13;
      }
      v25 = *(_DWORD *)(a1 + 16);
      if ( !v14 )
        v14 = v16;
      ++*(_QWORD *)a1;
      v23 = v25 + 1;
      if ( 4 * v23 < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(a1 + 20) - v23 > v10 >> 3 )
        {
LABEL_20:
          *(_DWORD *)(a1 + 16) = v23;
          if ( *v14 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v14 = v11;
          v18 = v14 + 1;
          v14[1] = 0;
          goto LABEL_16;
        }
        v36 = result;
        sub_2BFD020(a1, v10);
        v26 = *(_DWORD *)(a1 + 24);
        if ( v26 )
        {
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = 0;
          v30 = (v26 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v31 = 1;
          v23 = *(_DWORD *)(a1 + 16) + 1;
          result = v36;
          v14 = (_QWORD *)(v28 + 16LL * v30);
          v32 = *v14;
          if ( v11 != *v14 )
          {
            while ( v32 != -4096 )
            {
              if ( !v29 && v32 == -8192 )
                v29 = v14;
              v30 = v27 & (v31 + v30);
              v14 = (_QWORD *)(v28 + 16LL * v30);
              v32 = *v14;
              if ( v11 == *v14 )
                goto LABEL_20;
              ++v31;
            }
            if ( v29 )
              v14 = v29;
          }
          goto LABEL_20;
        }
LABEL_55:
        ++*(_DWORD *)(a1 + 16);
        goto LABEL_56;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    v35 = result;
    sub_2BFD020(a1, 2 * v10);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v23 = *(_DWORD *)(a1 + 16) + 1;
      result = v35;
      v14 = (_QWORD *)(v21 + 16LL * v22);
      v24 = *v14;
      if ( v11 != *v14 )
      {
        v33 = 1;
        v34 = 0;
        while ( v24 != -4096 )
        {
          if ( !v34 && v24 == -8192 )
            v34 = v14;
          v22 = v20 & (v33 + v22);
          v14 = (_QWORD *)(v21 + 16LL * v22);
          v24 = *v14;
          if ( v11 == *v14 )
            goto LABEL_20;
          ++v33;
        }
        if ( v34 )
          v14 = v34;
      }
      goto LABEL_20;
    }
    goto LABEL_55;
  }
  if ( v4 <= 0x36 )
  {
    if ( v4 > 0x34 )
      return sub_BCCE00(*(_QWORD **)(a1 + 40), 1u);
    if ( v4 != 12 )
LABEL_56:
      BUG();
    return sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
  }
  if ( v4 != 64 )
  {
    if ( v4 != 67 )
      goto LABEL_56;
    return sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
  }
  v5 = sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
  v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL) + 40LL);
  v7 = *(_DWORD *)(v6 + 32) <= 0x40u;
  v8 = *(_QWORD **)(v6 + 24);
  if ( !v7 )
    v8 = (_QWORD *)*v8;
  return *(_QWORD *)(*(_QWORD *)(v5 + 16) + 8LL * (unsigned int)v8);
}
