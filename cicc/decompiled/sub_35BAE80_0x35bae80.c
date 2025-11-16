// Function: sub_35BAE80
// Address: 0x35bae80
//
__int64 __fastcall sub_35BAE80(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v5; // rbx
  _QWORD *v6; // r14
  __int64 v7; // r8
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 result; // rax
  __int64 v22; // rcx
  __int64 v23; // rsi
  __int64 v24; // rcx
  _DWORD *v25; // rdi
  int v26; // eax
  __int64 v27; // rax
  int v28; // eax
  unsigned int v29; // [rsp+8h] [rbp-38h] BYREF
  int v30[13]; // [rsp+Ch] [rbp-34h] BYREF

  v5 = 48LL * a2;
  v6 = (_QWORD *)a1[19];
  if ( v6 )
  {
    v7 = 96LL * a3;
    v8 = v7 + *(_QWORD *)(*v6 + 160LL);
    v9 = v5 + *(_QWORD *)(*v6 + 208LL);
    v10 = *(_QWORD *)v9;
    v11 = *(_DWORD *)(v8 + 24);
    if ( a3 == *(_DWORD *)(v9 + 24) )
    {
      *(_DWORD *)(v8 + 24) = v11 - *(_DWORD *)(v10 + 16);
      v12 = *(_QWORD *)(v10 + 32);
    }
    else
    {
      *(_DWORD *)(v8 + 24) = v11 - *(_DWORD *)(v10 + 20);
      v12 = *(_QWORD *)(v10 + 24);
    }
    v13 = *(unsigned int *)(v8 + 20);
    if ( (_DWORD)v13 )
    {
      v14 = 0;
      do
      {
        v15 = v14++;
        *(_DWORD *)(*(_QWORD *)(v8 + 32) + 4 * v15) -= *(unsigned __int8 *)(v12 + v15);
        v13 = *(unsigned int *)(v8 + 20);
      }
      while ( (unsigned int)v13 > v14 );
    }
    if ( *(_QWORD *)(v7 + *(_QWORD *)(*v6 + 160LL) + 80) - *(_QWORD *)(v7 + *(_QWORD *)(*v6 + 160LL) + 72) == 12 )
    {
      v29 = a3;
      v30[0] = a3;
      v28 = *(_DWORD *)(*(_QWORD *)(*v6 + 160LL) + v7 + 16);
      switch ( v28 )
      {
        case 2:
          sub_35B9090(v6 + 7, (unsigned int *)v30);
          break;
        case 3:
          sub_35B9090(v6 + 1, (unsigned int *)v30);
          break;
        case 1:
          sub_35B9090(v6 + 13, (unsigned int *)v30);
          break;
      }
      sub_B99820((__int64)(v6 + 1), &v29);
      *(_DWORD *)(*(_QWORD *)(*v6 + 160LL) + 96LL * v29 + 16) = 3;
    }
    else if ( *(_DWORD *)(v8 + 16) == 1 )
    {
      if ( (unsigned int)v13 > *(_DWORD *)(v8 + 24)
        || (v30[0] = 0, v25 = *(_DWORD **)(v8 + 32), &v25[v13] != sub_35B8490(v25, (__int64)&v25[v13], v30)) )
      {
        v29 = a3;
        v30[0] = a3;
        v26 = *(_DWORD *)(*(_QWORD *)(*v6 + 160LL) + v7 + 16);
        switch ( v26 )
        {
          case 2:
            sub_35B9090(v6 + 7, (unsigned int *)v30);
            break;
          case 3:
            sub_35B9090(v6 + 1, (unsigned int *)v30);
            break;
          case 1:
            sub_35B9090(v6 + 13, (unsigned int *)v30);
            break;
        }
        sub_B99820((__int64)(v6 + 7), &v29);
        v27 = *(_QWORD *)(*v6 + 160LL) + 96LL * v29;
        *(_DWORD *)(v27 + 16) = 2;
        *(_BYTE *)(v27 + 64) = 1;
      }
    }
  }
  v16 = a1[26];
  v17 = a1[20];
  v18 = v16 + v5;
  if ( a3 == *(_DWORD *)(v18 + 20) )
  {
    v23 = *(_QWORD *)(v18 + 32);
    result = v17 + 96LL * a3;
    v24 = 48LL * *(unsigned int *)(*(_QWORD *)(result + 80) - 4LL) + v16;
    if ( a3 == *(_DWORD *)(v24 + 20) )
      *(_QWORD *)(v24 + 32) = v23;
    else
      *(_QWORD *)(v24 + 40) = v23;
    *(_DWORD *)(*(_QWORD *)(result + 72) + 4 * v23) = *(_DWORD *)(*(_QWORD *)(result + 80) - 4LL);
    *(_QWORD *)(result + 80) -= 4LL;
    *(_QWORD *)(v18 + 32) = -1;
  }
  else
  {
    v19 = *(_QWORD *)(v18 + 40);
    v20 = *(unsigned int *)(v18 + 24);
    result = v17 + 96 * v20;
    v22 = 48LL * *(unsigned int *)(*(_QWORD *)(result + 80) - 4LL) + v16;
    if ( (_DWORD)v20 == *(_DWORD *)(v22 + 20) )
      *(_QWORD *)(v22 + 32) = v19;
    else
      *(_QWORD *)(v22 + 40) = v19;
    *(_DWORD *)(*(_QWORD *)(result + 72) + 4 * v19) = *(_DWORD *)(*(_QWORD *)(result + 80) - 4LL);
    *(_QWORD *)(result + 80) -= 4LL;
    *(_QWORD *)(v18 + 40) = -1;
  }
  return result;
}
