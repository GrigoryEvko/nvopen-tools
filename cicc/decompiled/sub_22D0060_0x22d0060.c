// Function: sub_22D0060
// Address: 0x22d0060
//
void __fastcall sub_22D0060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
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
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // edi
  __int64 *v24; // r12
  __int64 v25; // rcx
  __int64 *v26; // r14
  int v27; // r9d
  __int64 v28; // rdx
  int v29; // r9d
  int v30; // r11d
  unsigned int v31; // eax
  _QWORD *v32; // r10
  unsigned int v33; // eax
  __int64 *v34; // rbx
  unsigned __int64 v35; // r15
  __int64 v36; // rdi
  unsigned int v37; // eax
  int v38; // r9d
  _QWORD *v39; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a1 + 88);
  v8 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v7 )
  {
    v10 = 1;
    for ( i = (v7 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                | ((unsigned __int64)(((unsigned int)&qword_4F8A320 >> 9) ^ ((unsigned int)&qword_4F8A320 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; i = (v7 - 1) & v13 )
    {
      v12 = v8 + 24LL * i;
      if ( *(__int64 **)v12 == &qword_4F8A320 && a2 == *(_QWORD *)(v12 + 8) )
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
          v39 = (_QWORD *)(v16 + v17);
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
            while ( v39 != v18 );
          }
        }
      }
    }
  }
LABEL_15:
  v21 = *(unsigned int *)(a1 + 56);
  v22 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v21 )
  {
    v23 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v24 = (__int64 *)(v22 + 32LL * v23);
    v25 = *v24;
    if ( a2 == *v24 )
    {
LABEL_17:
      if ( v24 != (__int64 *)(v22 + 32 * v21) )
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
              v31 = v29
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
              while ( 1 )
              {
                v32 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * v31);
                if ( v28 == *v32 && a2 == v32[1] )
                  break;
                if ( *v32 == -4096 )
                {
                  if ( v32[1] == -4096 )
                    goto LABEL_26;
                  v37 = v30 + v31;
                  ++v30;
                  v31 = v29 & v37;
                }
                else
                {
                  v33 = v30 + v31;
                  ++v30;
                  v31 = v29 & v33;
                }
              }
              *v32 = -8192;
              v32[1] = -8192;
              --*(_DWORD *)(a1 + 80);
              ++*(_DWORD *)(a1 + 84);
            }
LABEL_26:
            v26 = (__int64 *)*v26;
          }
          while ( v24 + 1 != v26 );
          v34 = (__int64 *)v24[1];
          while ( v26 != v34 )
          {
            v35 = (unsigned __int64)v34;
            v34 = (__int64 *)*v34;
            v36 = *(_QWORD *)(v35 + 24);
            if ( v36 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
            j_j___libc_free_0(v35);
          }
        }
        *v24 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
    }
    else
    {
      v38 = 1;
      while ( v25 != -4096 )
      {
        v23 = (v21 - 1) & (v38 + v23);
        v24 = (__int64 *)(v22 + 32LL * v23);
        v25 = *v24;
        if ( a2 == *v24 )
          goto LABEL_17;
        ++v38;
      }
    }
  }
}
