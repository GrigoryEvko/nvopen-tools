// Function: sub_1DF7A80
// Address: 0x1df7a80
//
__int64 __fastcall sub_1DF7A80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, _BYTE *a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rbx
  _QWORD *v10; // r13
  _QWORD *v11; // r11
  _QWORD *v12; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 v17; // rbx
  _QWORD *v18; // r12
  __int64 (*v20)(); // rdx
  int v21; // r15d
  __int64 v22; // rcx
  _QWORD *v23; // r11
  __int64 v24; // rdx
  __int64 v25; // r8
  int v26; // esi
  _DWORD *v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r10
  __int64 v31; // r9
  __int64 v32; // rdi
  __int64 v34; // [rsp+10h] [rbp-60h]
  _QWORD *v35; // [rsp+20h] [rbp-50h]
  _QWORD *v36; // [rsp+28h] [rbp-48h]
  unsigned int v37; // [rsp+30h] [rbp-40h]

  v6 = a1;
  v7 = a2;
  v8 = a3;
  v9 = a4;
  v10 = *(_QWORD **)(a3 + 24);
  v11 = *(_QWORD **)(a2 + 24);
  if ( v10 != v11 )
  {
    v12 = (_QWORD *)v10[8];
    if ( (unsigned int)((__int64)(v10[9] - (_QWORD)v12) >> 3) != 1 || (_QWORD *)*v12 != v11 )
      return 0;
    v14 = *((unsigned int *)a5 + 2);
    if ( (_DWORD)v14 )
    {
      v36 = v10;
      v15 = *a5;
      v16 = 4 * v14;
      v34 = v8;
      v17 = 0;
      v18 = *(_QWORD **)(a1 + 264);
      v35 = *(_QWORD **)(a2 + 24);
      do
      {
        v20 = *(__int64 (**)())(**(_QWORD **)(*v18 + 16LL) + 112LL);
        if ( v20 == sub_1D00B10 )
          BUG();
        v37 = *(_DWORD *)(v15 + v17);
        if ( *(_BYTE *)(*(_QWORD *)(v20() + 232) + 8LL * v37 + 4)
          && (*(_QWORD *)(v18[38] + 8LL * (v37 >> 6)) & (1LL << v37)) == 0 )
        {
          return 0;
        }
        v15 = *a5;
        v18 = *(_QWORD **)(a1 + 264);
        if ( (*(_QWORD *)(v18[38] + 8LL * (*(_DWORD *)(*a5 + v17) >> 6)) & (1LL << *(_DWORD *)(*a5 + v17))) != 0 )
          return 0;
        v17 += 4;
      }
      while ( v16 != v17 );
      v6 = a1;
      v10 = v36;
      v11 = v35;
      v7 = a2;
      v8 = v34;
      v9 = a4;
    }
  }
  if ( (*(_BYTE *)v7 & 4) == 0 )
  {
    while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
      v7 = *(_QWORD *)(v7 + 8);
  }
  v21 = *(_DWORD *)(v6 + 296);
  v22 = *(_QWORD *)(v7 + 8);
  v23 = v11 + 3;
  if ( !v21 )
    return 0;
LABEL_16:
  while ( v8 != v22 )
  {
    while ( (_QWORD *)v22 != v23 )
    {
      if ( (unsigned __int16)(**(_WORD **)(v22 + 16) - 12) > 1u )
      {
        if ( v22 == v8 )
          return 1;
        v24 = *(_QWORD *)(v22 + 32);
        v25 = v24 + 40LL * *(unsigned int *)(v22 + 40);
        if ( v24 == v25 )
        {
LABEL_42:
          --v21;
          if ( (*(_BYTE *)v22 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v22 + 46) & 8) != 0 )
              v22 = *(_QWORD *)(v22 + 8);
          }
          v22 = *(_QWORD *)(v22 + 8);
          if ( v21 )
            goto LABEL_16;
        }
        else
        {
          while ( *(_BYTE *)v24 != 12 )
          {
            if ( !*(_BYTE *)v24 && (*(_BYTE *)(v24 + 3) & 0x10) != 0 )
            {
              v26 = *(_DWORD *)(v24 + 8);
              if ( v26 >= 0 )
              {
                if ( *(_QWORD *)(v9 + 88) )
                {
                  v29 = *(_QWORD *)(v9 + 64);
                  if ( v29 )
                  {
                    v30 = v9 + 56;
                    do
                    {
                      while ( 1 )
                      {
                        v31 = *(_QWORD *)(v29 + 16);
                        v32 = *(_QWORD *)(v29 + 24);
                        if ( (unsigned int)v26 <= *(_DWORD *)(v29 + 32) )
                          break;
                        v29 = *(_QWORD *)(v29 + 24);
                        if ( !v32 )
                          goto LABEL_50;
                      }
                      v30 = v29;
                      v29 = *(_QWORD *)(v29 + 16);
                    }
                    while ( v31 );
LABEL_50:
                    if ( v30 != v9 + 56 && (unsigned int)v26 >= *(_DWORD *)(v30 + 32) )
                      return 0;
                  }
                }
                else
                {
                  v27 = *(_DWORD **)v9;
                  v28 = *(_QWORD *)v9 + 4LL * *(unsigned int *)(v9 + 8);
                  if ( *(_QWORD *)v9 != v28 )
                  {
                    while ( v26 != *v27 )
                    {
                      if ( (_DWORD *)v28 == ++v27 )
                        goto LABEL_41;
                    }
                    if ( v27 != (_DWORD *)v28 )
                      return 0;
                  }
                }
              }
            }
LABEL_41:
            v24 += 40;
            if ( v25 == v24 )
              goto LABEL_42;
          }
        }
        return 0;
      }
      if ( (*(_BYTE *)v22 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v22 + 46) & 8) != 0 )
          v22 = *(_QWORD *)(v22 + 8);
      }
      v22 = *(_QWORD *)(v22 + 8);
      if ( v8 == v22 )
        goto LABEL_21;
    }
LABEL_53:
    v23 = v10 + 3;
    *a6 = 1;
    v22 = v10[4];
  }
LABEL_21:
  if ( (_QWORD *)v22 == v23 )
    goto LABEL_53;
  return 1;
}
