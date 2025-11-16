// Function: sub_F03AD0
// Address: 0xf03ad0
//
__int64 __fastcall sub_F03AD0(unsigned int *a1, unsigned int a2)
{
  __int64 v4; // r9
  unsigned __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  int v10; // ecx
  unsigned int v11; // ecx
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  char v17; // si
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rax
  __int64 result; // rax
  int v23; // ecx
  __int64 v24; // r13
  __int64 v25; // rdx
  __int64 v26; // rax

  v4 = *(_QWORD *)a1;
  v5 = a1[2];
  v6 = *(unsigned int *)(*(_QWORD *)a1 + 12LL);
  if ( !(_DWORD)v5 )
  {
LABEL_10:
    v13 = 16;
    v12 = 0;
    v11 = 1;
    goto LABEL_11;
  }
  if ( (unsigned int)v6 >= *(_DWORD *)(v4 + 8) )
  {
    if ( a2 > (int)v5 - 1 )
    {
      v14 = a2 + 1;
      if ( v14 != v5 )
      {
        if ( v14 >= v5 )
        {
          v24 = v14 - v5;
          if ( v14 > a1[3] )
          {
            sub_C8D5F0((__int64)a1, a1 + 4, v14, 0x10u, v6, v4);
            v4 = *(_QWORD *)a1;
            v5 = a1[2];
          }
          v25 = v24;
          v26 = v4 + 16 * v5;
          do
          {
            if ( v26 )
            {
              *(_QWORD *)v26 = 0;
              *(_DWORD *)(v26 + 8) = 0;
              *(_DWORD *)(v26 + 12) = 0;
            }
            v26 += 16;
            --v25;
          }
          while ( v25 );
          v4 = *(_QWORD *)a1;
          a1[2] += v24;
          LODWORD(v6) = *(_DWORD *)(v4 + 12);
        }
        else
        {
          a1[2] = v14;
          LODWORD(v6) = *(_DWORD *)(v4 + 12);
        }
      }
    }
    goto LABEL_10;
  }
  v7 = a2 - 1;
  v8 = 16 * v7;
  v9 = v4 + 16 * v7;
  v10 = *(_DWORD *)(v9 + 12);
  if ( v10 )
  {
    *(_DWORD *)(v9 + 12) = v10 - 1;
    v15 = *(_QWORD *)a1;
    v13 = 16LL * a2;
    v16 = *(_QWORD *)(*(_QWORD *)(v15 + v8) + 8LL * *(unsigned int *)(v15 + v8 + 12));
    goto LABEL_16;
  }
  do
  {
    v11 = v7;
    v7 = (unsigned int)(v7 - 1);
    v12 = 16 * v7;
    LODWORD(v6) = *(_DWORD *)(v4 + 16 * v7 + 12);
  }
  while ( !(_DWORD)v6 );
  v4 += 16 * v7;
  v13 = 16LL * v11;
LABEL_11:
  *(_DWORD *)(v4 + 12) = v6 - 1;
  v15 = *(_QWORD *)a1;
  v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + v12) + 8LL * *(unsigned int *)(*(_QWORD *)a1 + v12 + 12));
  if ( a2 != v11 )
  {
    while ( 1 )
    {
      v17 = v16;
      v18 = v11;
      v19 = v16 & 0xFFFFFFFFFFFFFFC0LL;
      ++v11;
      v20 = v17 & 0x3F;
      v21 = v15 + 16 * v18;
      *(_QWORD *)v21 = v19;
      *(_DWORD *)(v21 + 8) = v20 + 1;
      *(_DWORD *)(v21 + 12) = v20;
      v16 = *(_QWORD *)(v19 + 8 * v20);
      if ( a2 == v11 )
        break;
      v15 = *(_QWORD *)a1;
    }
    v15 = *(_QWORD *)a1;
    v13 = 16LL * a2;
  }
LABEL_16:
  result = v15 + v13;
  v23 = v16 & 0x3F;
  *(_QWORD *)result = v16 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(result + 12) = v23;
  *(_DWORD *)(result + 8) = v23 + 1;
  return result;
}
