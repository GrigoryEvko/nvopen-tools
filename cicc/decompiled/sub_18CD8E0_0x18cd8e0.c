// Function: sub_18CD8E0
// Address: 0x18cd8e0
//
__int64 __fastcall sub_18CD8E0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v3; // r14
  __int64 v6; // rcx
  _QWORD *v7; // rdx
  _QWORD *i; // rcx
  __int64 v9; // rcx
  int v10; // edx
  int v11; // esi
  __int64 v12; // rdi
  int v13; // r10d
  __int64 *v14; // r9
  unsigned int v15; // edx
  __int64 *v16; // r12
  __int64 v17; // r8
  int v18; // edx
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  _QWORD *v27; // r15
  _QWORD *v28; // r12
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  _QWORD *v31; // r15
  _QWORD *v32; // r12
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  __int64 result; // rax

  v3 = a2;
  v6 = *(unsigned int *)(a1 + 24);
  v7 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)(a1 + 16) = 0;
  for ( i = &v7[24 * v6]; i != v7; v7 += 24 )
  {
    if ( v7 )
      *v7 = -8;
  }
  if ( a2 != a3 )
  {
    do
    {
      v9 = *v3;
      if ( *v3 != -8 && v9 != -16 )
      {
        v10 = *(_DWORD *)(a1 + 24);
        if ( !v10 )
        {
          MEMORY[0] = *v3;
          BUG();
        }
        v11 = v10 - 1;
        v12 = *(_QWORD *)(a1 + 8);
        v13 = 1;
        v14 = 0;
        v15 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v16 = (__int64 *)(v12 + 192LL * v15);
        v17 = *v16;
        if ( v9 != *v16 )
        {
          while ( v17 != -8 )
          {
            if ( v17 == -16 && !v14 )
              v14 = v16;
            v15 = v11 & (v13 + v15);
            v16 = (__int64 *)(v12 + 192LL * v15);
            v17 = *v16;
            if ( v9 == *v16 )
              goto LABEL_10;
            ++v13;
          }
          if ( v14 )
            v16 = v14;
        }
LABEL_10:
        *v16 = v9;
        *((_DWORD *)v16 + 2) = *((_DWORD *)v3 + 2);
        v18 = *((_DWORD *)v3 + 3);
        v16[4] = 0;
        v16[3] = 0;
        *((_DWORD *)v16 + 10) = 0;
        *((_DWORD *)v16 + 3) = v18;
        v16[2] = 1;
        v19 = v3[3];
        ++v3[2];
        v20 = v16[3];
        v16[3] = v19;
        LODWORD(v19) = *((_DWORD *)v3 + 8);
        v3[3] = v20;
        LODWORD(v20) = *((_DWORD *)v16 + 8);
        *((_DWORD *)v16 + 8) = v19;
        LODWORD(v19) = *((_DWORD *)v3 + 9);
        *((_DWORD *)v3 + 8) = v20;
        LODWORD(v20) = *((_DWORD *)v16 + 9);
        *((_DWORD *)v16 + 9) = v19;
        LODWORD(v19) = *((_DWORD *)v3 + 10);
        *((_DWORD *)v3 + 9) = v20;
        LODWORD(v20) = *((_DWORD *)v16 + 10);
        *((_DWORD *)v16 + 10) = v19;
        *((_DWORD *)v3 + 10) = v20;
        v16[6] = v3[6];
        v16[7] = v3[7];
        v16[8] = v3[8];
        v3[8] = 0;
        v3[7] = 0;
        v3[6] = 0;
        v16[11] = 0;
        v16[10] = 0;
        *((_DWORD *)v16 + 24) = 0;
        v16[9] = 1;
        v21 = v3[10];
        ++v3[9];
        v22 = v16[10];
        v16[10] = v21;
        LODWORD(v21) = *((_DWORD *)v3 + 22);
        v3[10] = v22;
        LODWORD(v22) = *((_DWORD *)v16 + 22);
        *((_DWORD *)v16 + 22) = v21;
        LODWORD(v21) = *((_DWORD *)v3 + 23);
        *((_DWORD *)v3 + 22) = v22;
        LODWORD(v22) = *((_DWORD *)v16 + 23);
        *((_DWORD *)v16 + 23) = v21;
        v23 = *((unsigned int *)v3 + 24);
        *((_DWORD *)v3 + 23) = v22;
        LODWORD(v22) = *((_DWORD *)v16 + 24);
        *((_DWORD *)v16 + 24) = v23;
        *((_DWORD *)v3 + 24) = v22;
        v16[13] = v3[13];
        v16[14] = v3[14];
        v16[15] = v3[15];
        v3[15] = 0;
        v3[14] = 0;
        v3[13] = 0;
        v16[16] = (__int64)(v16 + 18);
        v16[17] = 0x200000000LL;
        v24 = *((unsigned int *)v3 + 34);
        if ( (_DWORD)v24 )
          sub_18CD0F0((__int64)(v16 + 16), (char **)v3 + 16, v24, v23, v17, (int)v14);
        v16[20] = (__int64)(v16 + 22);
        v16[21] = 0x200000000LL;
        if ( *((_DWORD *)v3 + 42) )
          sub_18CD0F0((__int64)(v16 + 20), (char **)v3 + 20, (__int64)(v16 + 22), v23, v17, (int)v14);
        ++*(_DWORD *)(a1 + 16);
        v25 = v3[20];
        if ( (__int64 *)v25 != v3 + 22 )
          _libc_free(v25);
        v26 = v3[16];
        if ( (__int64 *)v26 != v3 + 18 )
          _libc_free(v26);
        v27 = (_QWORD *)v3[14];
        v28 = (_QWORD *)v3[13];
        if ( v27 != v28 )
        {
          do
          {
            v29 = v28[13];
            if ( v29 != v28[12] )
              _libc_free(v29);
            v30 = v28[6];
            if ( v30 != v28[5] )
              _libc_free(v30);
            v28 += 19;
          }
          while ( v27 != v28 );
          v28 = (_QWORD *)v3[13];
        }
        if ( v28 )
          j_j___libc_free_0(v28, v3[15] - (_QWORD)v28);
        j___libc_free_0(v3[10]);
        v31 = (_QWORD *)v3[7];
        v32 = (_QWORD *)v3[6];
        if ( v31 != v32 )
        {
          do
          {
            v33 = v32[13];
            if ( v33 != v32[12] )
              _libc_free(v33);
            v34 = v32[6];
            if ( v34 != v32[5] )
              _libc_free(v34);
            v32 += 19;
          }
          while ( v31 != v32 );
          v32 = (_QWORD *)v3[6];
        }
        if ( v32 )
          j_j___libc_free_0(v32, v3[8] - (_QWORD)v32);
        result = j___libc_free_0(v3[3]);
      }
      v3 += 24;
    }
    while ( a3 != v3 );
  }
  return result;
}
