// Function: sub_227B210
// Address: 0x227b210
//
void __fastcall sub_227B210(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  _QWORD *v12; // rbx
  _QWORD *v13; // rdi
  void (__fastcall *v14)(_QWORD *, __int64, __int64); // rcx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // edi
  __int64 *v19; // rbx
  __int64 v20; // rcx
  __int64 *v21; // r14
  int v22; // r9d
  __int64 v23; // rdx
  int v24; // r9d
  int v25; // r11d
  unsigned int v26; // eax
  _QWORD *v27; // r10
  unsigned int v28; // eax
  __int64 *v29; // r12
  unsigned __int64 v30; // r15
  __int64 v31; // rdi
  unsigned int v32; // eax
  int v33; // r9d
  _QWORD *v34; // [rsp+8h] [rbp-38h]

  v8 = sub_227B160(a1, (__int64)&unk_4F8A320, a2);
  if ( v8 )
  {
    v9 = *(_QWORD *)(v8 + 8);
    if ( v9 )
    {
      v10 = *(_QWORD *)(v9 + 1152);
      v11 = 32LL * *(unsigned int *)(v9 + 1160);
      v34 = (_QWORD *)(v10 + v11);
      if ( v10 != v10 + v11 )
      {
        v12 = *(_QWORD **)(v9 + 1152);
        do
        {
          v13 = v12;
          v14 = *(void (__fastcall **)(_QWORD *, __int64, __int64))(v12[3] & 0xFFFFFFFFFFFFFFF8LL);
          if ( (v12[3] & 2) == 0 )
            v13 = (_QWORD *)*v12;
          v12 += 4;
          v14(v13, a3, a4);
        }
        while ( v34 != v12 );
      }
    }
  }
  v15 = *(unsigned int *)(a1 + 56);
  v16 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v15 )
  {
    v17 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v19 = (__int64 *)(v16 + 32LL * v18);
    v20 = *v19;
    if ( a2 == *v19 )
    {
LABEL_10:
      if ( v19 != (__int64 *)(v16 + 32 * v15) )
      {
        v21 = (__int64 *)v19[1];
        if ( v19 + 1 != v21 )
        {
          do
          {
            v22 = *(_DWORD *)(a1 + 88);
            v23 = v21[2];
            if ( v22 )
            {
              v24 = v22 - 1;
              v25 = 1;
              v26 = v24
                  & (((0xBF58476D1CE4E5B9LL
                     * (v17 | ((unsigned __int64)(((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4)) << 32))) >> 31)
                   ^ (484763065 * v17));
              while ( 1 )
              {
                v27 = (_QWORD *)(*(_QWORD *)(a1 + 72) + 24LL * v26);
                if ( v23 == *v27 && a2 == v27[1] )
                  break;
                if ( *v27 == -4096 )
                {
                  if ( v27[1] == -4096 )
                    goto LABEL_19;
                  v32 = v25 + v26;
                  ++v25;
                  v26 = v24 & v32;
                }
                else
                {
                  v28 = v25 + v26;
                  ++v25;
                  v26 = v24 & v28;
                }
              }
              *v27 = -8192;
              v27[1] = -8192;
              --*(_DWORD *)(a1 + 80);
              ++*(_DWORD *)(a1 + 84);
            }
LABEL_19:
            v21 = (__int64 *)*v21;
          }
          while ( v19 + 1 != v21 );
          v29 = (__int64 *)v19[1];
          while ( v21 != v29 )
          {
            v30 = (unsigned __int64)v29;
            v29 = (__int64 *)*v29;
            v31 = *(_QWORD *)(v30 + 24);
            if ( v31 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v31 + 8LL))(v31);
            j_j___libc_free_0(v30);
          }
        }
        *v19 = -8192;
        --*(_DWORD *)(a1 + 48);
        ++*(_DWORD *)(a1 + 52);
      }
    }
    else
    {
      v33 = 1;
      while ( v20 != -4096 )
      {
        v18 = (v15 - 1) & (v33 + v18);
        v19 = (__int64 *)(v16 + 32LL * v18);
        v20 = *v19;
        if ( a2 == *v19 )
          goto LABEL_10;
        ++v33;
      }
    }
  }
}
