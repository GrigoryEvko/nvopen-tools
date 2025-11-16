// Function: sub_AE1EA0
// Address: 0xae1ea0
//
__int64 __fastcall sub_AE1EA0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  int v13; // ebx
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v18; // rdx
  _QWORD *v19; // r12
  _QWORD *v20; // rax
  _QWORD *v21; // rbx
  __int64 v22; // rdi

  v4 = *(_QWORD *)(a1 + 488);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 8);
    v6 = 16LL * *(unsigned int *)(v4 + 24);
    if ( *(_DWORD *)(v4 + 16) )
    {
      v19 = (_QWORD *)(v5 + v6);
      if ( v5 + v6 != v5 )
      {
        v20 = *(_QWORD **)(v4 + 8);
        while ( 1 )
        {
          v21 = v20;
          if ( *v20 != -4096 && *v20 != -8192 )
            break;
          v20 += 2;
          if ( v19 == v20 )
            goto LABEL_3;
        }
        if ( v19 != v20 )
        {
          do
          {
            v22 = v21[1];
            v21 += 2;
            _libc_free(v22, v6);
            if ( v21 == v19 )
              break;
            while ( *v21 == -8192 || *v21 == -4096 )
            {
              v21 += 2;
              if ( v19 == v21 )
                goto LABEL_38;
            }
          }
          while ( v21 != v19 );
LABEL_38:
          v5 = *(_QWORD *)(v4 + 8);
          v6 = 16LL * *(unsigned int *)(v4 + 24);
        }
      }
    }
LABEL_3:
    sub_C7D6A0(v5, v6, 8);
    j_j___libc_free_0(v4, 32);
  }
  *(_QWORD *)(a1 + 488) = 0;
  sub_2240AE0(a1 + 448, a2 + 448);
  *(_BYTE *)a1 = *(_BYTE *)a2;
  *(_DWORD *)(a1 + 4) = *(_DWORD *)(a2 + 4);
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_WORD *)(a1 + 16) = *(_WORD *)(a2 + 16);
  *(_WORD *)(a1 + 18) = *(_WORD *)(a2 + 18);
  *(_DWORD *)(a1 + 20) = *(_DWORD *)(a2 + 20);
  *(_DWORD *)(a1 + 24) = *(_DWORD *)(a2 + 24);
  if ( a1 + 32 != a2 + 32 )
  {
    v7 = *(_QWORD *)(a2 + 40);
    v8 = *(_QWORD *)(a1 + 40);
    if ( v7 <= v8 )
    {
      if ( v7 )
        memmove(*(void **)(a1 + 32), *(const void **)(a2 + 32), *(_QWORD *)(a2 + 40));
    }
    else
    {
      if ( v7 > *(_QWORD *)(a1 + 48) )
      {
        v18 = *(_QWORD *)(a2 + 40);
        v8 = 0;
        *(_QWORD *)(a1 + 40) = 0;
        sub_C8D290(a1 + 32, a1 + 56, v18, 1);
        v9 = *(_QWORD *)(a2 + 40);
      }
      else
      {
        v9 = *(_QWORD *)(a2 + 40);
        if ( v8 )
        {
          memmove(*(void **)(a1 + 32), *(const void **)(a2 + 32), *(_QWORD *)(a1 + 40));
          v9 = *(_QWORD *)(a2 + 40);
        }
      }
      v10 = *(_QWORD *)(a2 + 32);
      if ( v10 + v8 != v9 + v10 )
        memcpy((void *)(v8 + *(_QWORD *)(a1 + 32)), (const void *)(v10 + v8), v9 - v8);
    }
    *(_QWORD *)(a1 + 40) = v7;
  }
  sub_AE1280(a1 + 64, a2 + 64);
  sub_AE1280(a1 + 128, a2 + 128);
  sub_AE1280(a1 + 176, a2 + 176);
  if ( a1 + 272 != a2 + 272 )
  {
    v11 = *(unsigned int *)(a2 + 280);
    v12 = *(unsigned int *)(a1 + 280);
    v13 = *(_DWORD *)(a2 + 280);
    if ( v11 <= v12 )
    {
      if ( *(_DWORD *)(a2 + 280) )
        memmove(*(void **)(a1 + 272), *(const void **)(a2 + 272), 20 * v11);
    }
    else
    {
      if ( v11 > *(unsigned int *)(a1 + 284) )
      {
        v14 = 0;
        *(_DWORD *)(a1 + 280) = 0;
        sub_C8D5F0(a1 + 272, a1 + 288, v11, 20);
        v11 = *(unsigned int *)(a2 + 280);
      }
      else
      {
        v14 = 20 * v12;
        if ( *(_DWORD *)(a1 + 280) )
        {
          memmove(*(void **)(a1 + 272), *(const void **)(a2 + 272), 20 * v12);
          v11 = *(unsigned int *)(a2 + 280);
        }
      }
      v15 = *(_QWORD *)(a2 + 272);
      v16 = 20 * v11;
      if ( v15 + v14 != v16 + v15 )
        memcpy((void *)(v14 + *(_QWORD *)(a1 + 272)), (const void *)(v15 + v14), v16 - v14);
    }
    *(_DWORD *)(a1 + 280) = v13;
  }
  *(_BYTE *)(a1 + 480) = *(_BYTE *)(a2 + 480);
  *(_BYTE *)(a1 + 481) = *(_BYTE *)(a2 + 481);
  return a1;
}
