// Function: sub_878A90
// Address: 0x878a90
//
__int64 __fastcall sub_878A90(__int64 a1)
{
  int v2; // r8d
  _QWORD *v3; // r10
  unsigned int v4; // ecx
  __int64 v5; // rsi
  int v6; // edi
  __int64 v7; // rax
  __int64 v8; // rdx
  int v9; // ebx
  int v10; // r12d
  unsigned int v11; // r11d
  int *v12; // rdx
  unsigned int v13; // edx
  int v14; // r8d
  __int64 v15; // rdi
  unsigned int v16; // esi
  int *v17; // rcx
  int v18; // edx
  int v19; // ebx
  int *v20; // rax
  int v21; // r13d
  int v22; // r12d
  __int64 *v23; // rdx
  __int64 v24; // rbx
  __int64 v25; // r14
  unsigned int v26; // edx
  int *v27; // rcx
  __int64 result; // rax
  __int64 v29; // rbx

  v2 = *(_DWORD *)(a1 + 28);
  v3 = qword_4F5FED8;
  v4 = *((_DWORD *)qword_4F5FED8 + 2);
  v5 = *qword_4F5FED8;
  v6 = *(_DWORD *)(a1 + 24);
  v7 = v4 & (v2 + 31 * (v6 + 527));
  v8 = *qword_4F5FED8 + 16 * v7;
  v9 = *(_DWORD *)v8;
  v10 = *(_DWORD *)(v8 + 4);
  if ( v6 == *(_DWORD *)v8 && v2 == v10 )
  {
    if ( *(_QWORD *)(v8 + 8) )
    {
LABEL_10:
      *(_QWORD *)v8 = 0;
      v13 = v7 + 1;
      if ( *(_QWORD *)(v5 + 16LL * (((_DWORD)v7 + 1) & v4)) )
      {
        v14 = *((_DWORD *)v3 + 2);
        v15 = *v3;
        v16 = v14 & v13;
        v17 = (int *)(*v3 + 16LL * (v14 & v13));
        v18 = *v17;
        v19 = v17[1];
        while ( 1 )
        {
          v26 = v14 & (v19 + 31 * (v18 + 527));
          v27 = (int *)(v15 + 16LL * (v14 & (v16 + 1)));
          if ( v26 <= (unsigned int)v7 && (v16 < v26 || v16 > (unsigned int)v7) || v16 > (unsigned int)v7 && v16 < v26 )
          {
            v20 = (int *)(v15 + 16 * v7);
            v21 = *v20;
            v22 = v20[1];
            v23 = (__int64 *)(v15 + 16LL * v16);
            if ( *(_QWORD *)v20 )
            {
              v24 = *v23;
              v25 = *((_QWORD *)v20 + 1);
              *(_QWORD *)v20 = *v23;
              if ( v24 )
                *((_QWORD *)v20 + 1) = v23[1];
              *(_DWORD *)v23 = v21;
              *((_DWORD *)v23 + 1) = v22;
              v23[1] = v25;
            }
            else
            {
              v29 = *v23;
              *(_QWORD *)v20 = *v23;
              if ( v29 )
                *((_QWORD *)v20 + 1) = v23[1];
              *v23 = 0;
            }
            v18 = *v27;
            v19 = v27[1];
            if ( !*(_QWORD *)v27 )
              break;
            v7 = v16;
          }
          else
          {
            v18 = *v27;
            v19 = v27[1];
            if ( !*(_QWORD *)v27 )
              break;
          }
          v16 = v14 & (v16 + 1);
        }
      }
      --*((_DWORD *)v3 + 3);
    }
  }
  else
  {
    v11 = v7;
    while ( v9 | v10 )
    {
      v11 = v4 & (v11 + 1);
      v12 = (int *)(v5 + 16LL * v11);
      v9 = *v12;
      v10 = v12[1];
      if ( v6 == *v12 && v2 == v10 )
      {
        if ( !*((_QWORD *)v12 + 1) )
          break;
        do
        {
          do
          {
            v7 = v4 & ((_DWORD)v7 + 1);
            v8 = v5 + 16LL * (unsigned int)v7;
          }
          while ( *(_DWORD *)(v8 + 4) != v2 );
        }
        while ( *(_DWORD *)v8 != v6 );
        goto LABEL_10;
      }
    }
  }
  result = qword_4F5FFE8;
  *(_QWORD *)a1 = qword_4F5FFE8;
  qword_4F5FFE8 = a1;
  return result;
}
