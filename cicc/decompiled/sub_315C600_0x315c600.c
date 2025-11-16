// Function: sub_315C600
// Address: 0x315c600
//
__int64 __fastcall sub_315C600(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // r8
  unsigned int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 *v16; // rbx
  __int64 *v17; // r15
  __int64 *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // r8
  unsigned int v22; // edx
  __int64 v23; // rcx
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 *v26; // rbx
  __int64 *v27; // r13
  __int64 *v28; // rsi
  int v30; // ecx
  int v31; // ecx

  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0x400000000LL;
  v9 = *(unsigned int *)(a2 + 40);
  v10 = *(_QWORD *)(a2 + 24);
  if ( (_DWORD)v9 )
  {
    v11 = (unsigned int)(v9 - 1);
    v12 = v11 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v13 = v10 + 88LL * v12;
    v14 = *(_QWORD *)v13;
    if ( a3 == *(_QWORD *)v13 )
    {
LABEL_3:
      v15 = 5 * v9;
      if ( v13 != v10 + 88 * v9 )
      {
        v16 = *(__int64 **)(v13 + 40);
        v17 = &v16[*(unsigned int *)(v13 + 48)];
        while ( v17 != v16 )
        {
          v18 = v16++;
          sub_31599A0(a1, v18, v15, v13, v11, a6);
        }
      }
    }
    else
    {
      v30 = 1;
      while ( v14 != -4096 )
      {
        a6 = (unsigned int)(v30 + 1);
        v12 = v11 & (v30 + v12);
        v13 = v10 + 88LL * v12;
        v14 = *(_QWORD *)v13;
        if ( a3 == *(_QWORD *)v13 )
          goto LABEL_3;
        v30 = a6;
      }
    }
  }
  v19 = *(unsigned int *)(a2 + 72);
  v20 = *(_QWORD *)(a2 + 56);
  if ( (_DWORD)v19 )
  {
    v21 = (unsigned int)(v19 - 1);
    v22 = v21 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v23 = v20 + 88LL * v22;
    v24 = *(_QWORD *)v23;
    if ( a3 == *(_QWORD *)v23 )
    {
LABEL_8:
      v25 = 5 * v19;
      if ( v23 != v20 + 88 * v19 )
      {
        v26 = *(__int64 **)(v23 + 40);
        v27 = &v26[*(unsigned int *)(v23 + 48)];
        while ( v27 != v26 )
        {
          v28 = v26++;
          sub_31599A0(a1, v28, v25, v23, v21, a6);
        }
      }
    }
    else
    {
      v31 = 1;
      while ( v24 != -4096 )
      {
        a6 = (unsigned int)(v31 + 1);
        v22 = v21 & (v31 + v22);
        v23 = v20 + 88LL * v22;
        v24 = *(_QWORD *)v23;
        if ( a3 == *(_QWORD *)v23 )
          goto LABEL_8;
        v31 = a6;
      }
    }
  }
  return a1;
}
