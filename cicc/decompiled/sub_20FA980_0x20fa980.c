// Function: sub_20FA980
// Address: 0x20fa980
//
void __fastcall sub_20FA980(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v7; // r12
  __int64 *v8; // r13
  __int64 v9; // r15
  int v10; // eax
  __int64 v11; // rdx
  __int64 *v12; // rbx
  int v13; // esi
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // r13
  __int64 v24; // r12
  __int64 v25; // rax
  int v26; // eax
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v7 = 0;
  v8 = *(__int64 **)a2;
  v9 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( v9 != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v10 = *(_DWORD *)(a3 + 24);
      v11 = *v8;
      v12 = 0;
      if ( v10 )
      {
        v13 = v10 - 1;
        v14 = *(_QWORD *)(a3 + 8);
        v15 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = (__int64 *)(v14 + 16LL * v15);
        a5 = *v16;
        if ( v11 == *v16 )
        {
LABEL_4:
          v12 = (__int64 *)v16[1];
        }
        else
        {
          v26 = 1;
          while ( a5 != -8 )
          {
            LODWORD(a6) = v26 + 1;
            v15 = v13 & (v26 + v15);
            v16 = (__int64 *)(v14 + 16LL * v15);
            a5 = *v16;
            if ( v11 == *v16 )
              goto LABEL_4;
            v26 = a6;
          }
          v12 = 0;
        }
      }
      if ( v12 != v7
        && v7
        && (*((_DWORD *)v7 + 44) >= *((_DWORD *)v12 + 44) || *((_DWORD *)v7 + 45) <= *((_DWORD *)v12 + 45)) )
      {
        do
        {
          a6 = v7[21];
          a5 = v7[20];
          v17 = *((unsigned int *)v7 + 22);
          if ( (unsigned int)v17 >= *((_DWORD *)v7 + 23) )
          {
            v27 = v7[20];
            v28 = v7[21];
            sub_16CD150((__int64)(v7 + 10), v7 + 12, 0, 16, a5, a6);
            v17 = *((unsigned int *)v7 + 22);
            a5 = v27;
            a6 = v28;
          }
          v18 = (_QWORD *)(v7[10] + 16 * v17);
          *v18 = a6;
          v18[1] = a5;
          v7[21] = 0;
          ++*((_DWORD *)v7 + 22);
          v7[20] = 0;
          v7 = (__int64 *)*v7;
        }
        while ( v7
             && v12 != v7
             && (*((_DWORD *)v7 + 44) >= *((_DWORD *)v12 + 44) || *((_DWORD *)v7 + 45) <= *((_DWORD *)v12 + 45)) );
        v11 = *v8;
      }
      v19 = v12;
      do
      {
        while ( v19[21] )
        {
          v19 = (_QWORD *)*v19;
          if ( !v19 )
            goto LABEL_20;
        }
        v19[21] = v11;
        v19 = (_QWORD *)*v19;
      }
      while ( v19 );
LABEL_20:
      v20 = v8[1];
      v21 = v12;
      do
      {
        v21[20] = v20;
        v21 = (_QWORD *)*v21;
      }
      while ( v21 );
      v8 += 2;
      if ( (__int64 *)v9 == v8 )
        break;
      v7 = v12;
    }
    while ( v12 )
    {
      v23 = v12[21];
      v24 = v12[20];
      v25 = *((unsigned int *)v12 + 22);
      if ( (unsigned int)v25 >= *((_DWORD *)v12 + 23) )
      {
        sub_16CD150((__int64)(v12 + 10), v12 + 12, 0, 16, a5, a6);
        v25 = *((unsigned int *)v12 + 22);
      }
      v22 = (_QWORD *)(v12[10] + 16 * v25);
      *v22 = v23;
      v22[1] = v24;
      v12[21] = 0;
      ++*((_DWORD *)v12 + 22);
      v12[20] = 0;
      v12 = (__int64 *)*v12;
    }
  }
}
