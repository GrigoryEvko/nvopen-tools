// Function: sub_2FDEB30
// Address: 0x2fdeb30
//
bool __fastcall sub_2FDEB30(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // r15d
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  int *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r13
  int v14; // eax
  __int64 *v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rax

  v7 = *(_QWORD *)(a2 + 48);
  v8 = *(_DWORD *)(a3 + 8);
  v9 = v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_7;
  if ( (v7 & 7) == 0 )
  {
    *(_QWORD *)(a2 + 48) = v9;
    v10 = a2 + 48;
    v7 &= 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_4;
  }
  v10 = v9 + 16;
  if ( (v7 & 7) != 3 )
  {
LABEL_7:
    v11 = (int *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      LODWORD(v12) = v8;
      return v8 != (_DWORD)v12;
    }
    v10 = 0;
    goto LABEL_9;
  }
LABEL_4:
  v11 = (int *)(v7 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v12 = *(unsigned int *)(a3 + 8);
    v13 = 0;
    do
    {
LABEL_12:
      v15 = *(__int64 **)v10;
      if ( (*(_BYTE *)(*(_QWORD *)v10 + 32LL) & 2) != 0 )
      {
        v16 = *v15;
        if ( *v15 )
        {
          if ( (v16 & 4) != 0 )
          {
            v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v17 )
            {
              if ( *(_DWORD *)(v17 + 8) == 4 )
              {
                if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
                {
                  sub_C8D5F0(a3, (const void *)(a3 + 16), v12 + 1, 8u, v12 + 1, a6);
                  v12 = *(unsigned int *)(a3 + 8);
                }
                *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v15;
                v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
                *(_DWORD *)(a3 + 8) = v12;
              }
            }
          }
        }
      }
      v10 += 8LL;
    }
    while ( v10 != v13 );
    return v8 != (_DWORD)v12;
  }
LABEL_9:
  v14 = v7 & 7;
  if ( v14 )
  {
    v13 = 0;
    if ( v14 == 3 )
      v13 = (__int64)&v11[2 * *v11 + 4];
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v11;
    v13 = a2 + 56;
  }
  v12 = *(unsigned int *)(a3 + 8);
  if ( v13 != v10 )
    goto LABEL_12;
  return v8 != (_DWORD)v12;
}
