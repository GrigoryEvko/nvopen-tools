// Function: sub_BBB260
// Address: 0xbbb260
//
__int64 __fastcall sub_BBB260(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rsi
  int v10; // r10d
  unsigned int i; // eax
  __int64 v12; // rdi
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r12
  _QWORD *v18; // r12
  _QWORD *v19; // rdi
  void (__fastcall *v20)(_QWORD *, __int64, __int64); // rcx
  __int64 result; // rax
  __int64 v22; // rdx
  unsigned int v23; // edi
  __int64 *v24; // r12
  __int64 v25; // rcx
  __int64 *v26; // r14
  int v27; // r9d
  __int64 v28; // rdx
  int v29; // r9d
  int v30; // r11d
  _QWORD *v31; // r10
  unsigned int v32; // eax
  __int64 *j; // rbx
  __int64 *v34; // r15
  __int64 v35; // rdi
  unsigned int v36; // eax
  int v37; // r9d
  _QWORD *v38; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 88);
  v8 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v7 )
  {
    v10 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F8A320 >> 9) ^ ((unsigned int)&unk_4F8A320 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v7 - 1) & v13 )
    {
      v12 = v8 + 24LL * i;
      if ( *(_UNKNOWN **)v12 == &unk_4F8A320 && a2 == *(_QWORD *)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
        goto LABEL_15;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != v8 + 24 * v7 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
      if ( v14 )
      {
        v15 = *(_QWORD *)(v14 + 8);
        if ( v15 )
        {
          v16 = *(_QWORD *)(v15 + 1152);
          v17 = 32LL * *(unsigned int *)(v15 + 1160);
          v38 = (_QWORD *)(v16 + v17);
          if ( v16 != v16 + v17 )
          {
            v18 = *(_QWORD **)(v15 + 1152);
            do
            {
              v19 = v18;
              v20 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(v18[3] & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v18[3] & 2) == 0 )
                v19 = (_QWORD *)*v18;
              v18 += 4;
              v20(v19, a3, a4);
            }
            while ( v38 != v18 );
          }
        }
      }
    }
  }
LABEL_15:
  result = *(unsigned int *)(a1 + 56);
  v22 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)result )
  {
    v23 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v24 = (__int64 *)(v22 + 32LL * v23);
    v25 = *v24;
    if ( a2 == *v24 )
    {
LABEL_17:
      result = v22 + 32 * result;
      if ( v24 != (__int64 *)result )
      {
        v26 = (__int64 *)v24[1];
        if ( v24 + 1 != v26 )
        {
          do
          {
            v27 = *(_DWORD *)(a1 + 88);
            v28 = v26[2];
            if ( v27 )
            {
              v29 = v27 - 1;
              v30 = 1;
              result = v29
                     & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                      * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                                       | ((unsigned __int64)(((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)) << 32))) >> 31)
                      ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
              while ( 1 )
              {
                v31 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * (unsigned int)result);
                if ( v28 == *v31 && a2 == v31[1] )
                  break;
                if ( *v31 == -4096 )
                {
                  if ( v31[1] == -4096 )
                    goto LABEL_26;
                  v36 = v30 + result;
                  ++v30;
                  result = v29 & v36;
                }
                else
                {
                  v32 = v30 + result;
                  ++v30;
                  result = v29 & v32;
                }
              }
              *v31 = -8192;
              v31[1] = -8192;
              --*(_DWORD *)(a1 + 80);
              ++*(_DWORD *)(a1 + 84);
            }
LABEL_26:
            v26 = (__int64 *)*v26;
          }
          while ( v24 + 1 != v26 );
          for ( j = (__int64 *)v24[1]; v26 != j; result = j_j___libc_free_0(v34, 32) )
          {
            v34 = j;
            j = (__int64 *)*j;
            v35 = v34[3];
            if ( v35 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
          }
        }
        *v24 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
    }
    else
    {
      v37 = 1;
      while ( v25 != -4096 )
      {
        v23 = (result - 1) & (v37 + v23);
        v24 = (__int64 *)(v22 + 32LL * v23);
        v25 = *v24;
        if ( a2 == *v24 )
          goto LABEL_17;
        ++v37;
      }
    }
  }
  return result;
}
