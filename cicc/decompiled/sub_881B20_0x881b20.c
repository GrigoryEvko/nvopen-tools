// Function: sub_881B20
// Address: 0x881b20
//
__int64 __fastcall sub_881B20(unsigned __int8 *a1, __int64 a2, int a3)
{
  unsigned int (__fastcall *v4)(_QWORD, __int64); // r13
  _QWORD *j; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int v8; // r14d
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v12; // rdx
  int v13; // edi
  int v14; // eax
  __int64 v15; // rax
  unsigned int *v16; // rax
  __int64 v17; // rdx
  unsigned int i; // ebx
  __int64 v19; // r13
  char *v20; // r12
  unsigned int v21; // r10d
  unsigned int v22; // edi
  _QWORD *v23; // rsi
  char *v24; // rax
  __int64 v25; // [rsp+0h] [rbp-40h]

  v4 = (unsigned int (__fastcall *)(_QWORD, __int64))off_4A51EC0[a1[1]];
  v8 = ((__int64 (__fastcall *)(__int64))off_4A51EC0[*a1])(a2);
  v9 = v8 % *((_DWORD *)a1 + 2);
  v25 = 8 * v9;
  v10 = *(_QWORD *)(*((_QWORD *)a1 + 2) + 8LL * (unsigned int)v9);
  if ( v10 )
  {
    while ( *(_DWORD *)(v10 + 16) != v8 || !v4(*(_QWORD *)(v10 + 8), a2) )
    {
      v10 = *(_QWORD *)v10;
      if ( !v10 )
        goto LABEL_7;
    }
    return v10 + 8;
  }
LABEL_7:
  if ( a3 )
  {
    v12 = *((unsigned int *)a1 + 2);
    v13 = *((_DWORD *)a1 + 1);
    v14 = *((_DWORD *)a1 + 3) + 1;
    *((_DWORD *)a1 + 3) = v14;
    if ( (double)v14 / (double)(int)v12 > 1.0 )
    {
      v16 = (unsigned int *)&unk_3C1F2C4;
      v17 = (unsigned int)(4 * v12);
      for ( i = 1; ; i = *v16++ )
      {
        if ( (unsigned int)v17 <= i )
        {
          v19 = 8LL * i;
          goto LABEL_15;
        }
        if ( v16 == (unsigned int *)&unk_3C1F3A8 )
          break;
      }
      v19 = 0;
      i = 0;
LABEL_15:
      v20 = (char *)sub_823020(v13, v19, v17, (__int64)&unk_3C1F3A8, v6, v7);
      memset(v20, 0, v19);
      v21 = *((_DWORD *)a1 + 2);
      v22 = 0;
      if ( v21 )
      {
        do
        {
          for ( j = *(_QWORD **)(*((_QWORD *)a1 + 2) + 8LL * v22); j; *(_QWORD *)v24 = v23 )
          {
            v23 = j;
            j = (_QWORD *)*j;
            v24 = &v20[8 * (*((_DWORD *)v23 + 4) % i)];
            *v23 = *(_QWORD *)v24;
          }
          ++v22;
        }
        while ( *((_DWORD *)a1 + 2) > v22 );
      }
      v13 = *((_DWORD *)a1 + 1);
      *((_DWORD *)a1 + 2) = i;
      if ( v13 == -1 )
      {
        sub_822B90(*((_QWORD *)a1 + 2), 8LL * v21);
        i = *((_DWORD *)a1 + 2);
        v13 = *((_DWORD *)a1 + 1);
      }
      *((_QWORD *)a1 + 2) = v20;
      v12 = v8 % i;
      v25 = 8LL * (unsigned int)v12;
    }
    v15 = sub_823020(v13, 24, v12, (__int64)j, v6, v7);
    *(_QWORD *)v15 = 0;
    v10 = v15;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)v15 = *(_QWORD *)(*((_QWORD *)a1 + 2) + v25);
    *(_QWORD *)(*((_QWORD *)a1 + 2) + v25) = v15;
    *(_DWORD *)(v15 + 16) = v8;
    return v10 + 8;
  }
  return 0;
}
