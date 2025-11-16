// Function: sub_2BFE840
// Address: 0x2bfe840
//
__int64 __fastcall sub_2BFE840(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // rcx
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // r13
  __int64 v8; // rbx
  __int64 v9; // r8
  int v10; // r11d
  _QWORD *v11; // rcx
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rax
  int v17; // eax
  int v18; // edi
  __int64 v19; // r8
  unsigned int v20; // esi
  int v21; // edx
  __int64 v22; // rax
  int v23; // eax
  int v24; // eax
  int v25; // esi
  __int64 v26; // rdi
  _QWORD *v27; // r8
  unsigned int v28; // r14d
  int v29; // r9d
  __int64 v30; // rax
  int v31; // r10d
  _QWORD *v32; // r9
  __int64 v33[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(unsigned __int8 **)(a2 + 136);
  if ( (unsigned __int8)(*v4 - 42) <= 0x11u )
  {
    v5 = sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
    v6 = *(_DWORD *)(a1 + 24);
    v7 = v5;
    v8 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
    if ( v6 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      v10 = 1;
      v11 = 0;
      v12 = (v6 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v13 = (_QWORD *)(v9 + 16LL * v12);
      v14 = *v13;
      if ( v8 == *v13 )
      {
LABEL_6:
        v15 = v13 + 1;
LABEL_7:
        *v15 = v7;
        return v7;
      }
      while ( v14 != -4096 )
      {
        if ( !v11 && v14 == -8192 )
          v11 = v13;
        v12 = (v6 - 1) & (v10 + v12);
        v13 = (_QWORD *)(v9 + 16LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          goto LABEL_6;
        ++v10;
      }
      if ( !v11 )
        v11 = v13;
      v23 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v21 = v23 + 1;
      if ( 4 * (v23 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 20) - v21 > v6 >> 3 )
        {
LABEL_13:
          *(_DWORD *)(a1 + 16) = v21;
          if ( *v11 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v11 = v8;
          v15 = v11 + 1;
          v11[1] = 0;
          goto LABEL_7;
        }
        sub_2BFD020(a1, v6);
        v24 = *(_DWORD *)(a1 + 24);
        if ( v24 )
        {
          v25 = v24 - 1;
          v26 = *(_QWORD *)(a1 + 8);
          v27 = 0;
          v28 = (v24 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v29 = 1;
          v21 = *(_DWORD *)(a1 + 16) + 1;
          v11 = (_QWORD *)(v26 + 16LL * v28);
          v30 = *v11;
          if ( v8 != *v11 )
          {
            while ( v30 != -4096 )
            {
              if ( !v27 && v30 == -8192 )
                v27 = v11;
              v28 = v25 & (v29 + v28);
              v11 = (_QWORD *)(v26 + 16LL * v28);
              v30 = *v11;
              if ( v8 == *v11 )
                goto LABEL_13;
              ++v29;
            }
            if ( v27 )
              v11 = v27;
          }
          goto LABEL_13;
        }
LABEL_52:
        ++*(_DWORD *)(a1 + 16);
LABEL_53:
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_2BFD020(a1, 2 * v6);
    v17 = *(_DWORD *)(a1 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(a1 + 8);
      v20 = (v17 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v21 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (_QWORD *)(v19 + 16LL * v20);
      v22 = *v11;
      if ( v8 != *v11 )
      {
        v31 = 1;
        v32 = 0;
        while ( v22 != -4096 )
        {
          if ( !v32 && v22 == -8192 )
            v32 = v11;
          v20 = v18 & (v31 + v20);
          v11 = (_QWORD *)(v19 + 16LL * v20);
          v22 = *v11;
          if ( v8 == *v11 )
            goto LABEL_13;
          ++v31;
        }
        if ( v32 )
          v11 = v32;
      }
      goto LABEL_13;
    }
    goto LABEL_52;
  }
  if ( (unsigned int)*v4 - 67 > 0xC )
  {
    switch ( *v4 )
    {
      case ')':
      case '?':
      case '`':
        return sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
      case '<':
      case '=':
      case ']':
        return *((_QWORD *)v4 + 1);
      case '>':
        return sub_BCB120(*(_QWORD **)(a1 + 40));
      case 'R':
      case 'S':
        return sub_BCCE00(*(_QWORD **)(a1 + 40), 1u);
      case 'U':
        return **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 48)
                                                               + 8LL
                                                               * (*(_DWORD *)(a2 + 56)
                                                                - (1
                                                                 - ((unsigned int)(*(_BYTE *)(a2 + 161) == 0)
                                                                  - 1))))
                                                   + 40LL)
                                       + 24LL)
                           + 16LL);
      case 'V':
        v7 = sub_2BFD6A0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
        v33[0] = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 16LL);
        v15 = sub_2BFD450(a1, v33);
        goto LABEL_7;
      default:
        goto LABEL_53;
    }
  }
  return *((_QWORD *)v4 + 1);
}
