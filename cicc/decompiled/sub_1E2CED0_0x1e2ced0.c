// Function: sub_1E2CED0
// Address: 0x1e2ced0
//
void __fastcall sub_1E2CED0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rax
  int v5; // r10d
  unsigned int v6; // edx
  __int64 v7; // r12
  __int64 v8; // rdi
  __int64 *v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi

  v14 = *(_QWORD *)(a1 + 1736);
  if ( v14 )
  {
    v3 = *(unsigned int *)(v14 + 88);
    if ( (_DWORD)v3 )
    {
      v5 = 1;
      v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v7 = v14;
      v8 = *(_QWORD *)(v14 + 72);
      v9 = (__int64 *)(v8 + 32LL * v6);
      v10 = *v9;
      if ( a2 == *v9 )
      {
LABEL_4:
        if ( v9 != (__int64 *)(v8 + 32 * v3) )
        {
          v11 = *a3;
          v12 = a3[1];
          v13 = a3[2];
          *a3 = v9[1];
          a3[1] = v9[2];
          a3[2] = v9[3];
          v9[1] = v11;
          v9[2] = v12;
          v9[3] = v13;
          if ( v11 )
            j_j___libc_free_0(v11, v13 - v11);
          *v9 = -16;
          --*(_DWORD *)(v7 + 80);
          ++*(_DWORD *)(v7 + 84);
        }
      }
      else
      {
        while ( v10 != -8 )
        {
          v6 = (v3 - 1) & (v5 + v6);
          v9 = (__int64 *)(v8 + 32LL * v6);
          v10 = *v9;
          if ( a2 == *v9 )
            goto LABEL_4;
          ++v5;
        }
      }
    }
    else
    {
      nullsub_2020();
    }
  }
}
