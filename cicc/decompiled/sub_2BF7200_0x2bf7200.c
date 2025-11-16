// Function: sub_2BF7200
// Address: 0x2bf7200
//
void __fastcall sub_2BF7200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // r12
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  int v11; // r15d
  _QWORD *v12; // rbx
  _QWORD *v13; // rax
  _QWORD *i; // rcx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rdx
  int v22; // eax
  int v23; // [rsp-50h] [rbp-50h]
  unsigned __int64 v24; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v7 = (_QWORD *)(a2 + 16);
    v8 = *(_QWORD **)a2;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v9 = *(unsigned int *)(a2 + 8);
      v10 = *(unsigned int *)(a1 + 8);
      v11 = *(_DWORD *)(a2 + 8);
      if ( v9 <= v10 )
      {
        if ( *(_DWORD *)(a2 + 8) )
        {
          v16 = *(_QWORD *)a1;
          v17 = a2 + 40 * v9 + 16;
          do
          {
            v18 = v7[4];
            v7 += 5;
            v16 += 40LL;
            *(_QWORD *)(v16 - 8) = v18;
            *(_QWORD *)(v16 - 24) = *(v7 - 3);
            *(_QWORD *)(v16 - 16) = *(v7 - 2);
            *(_QWORD *)(v16 - 40) = *(v7 - 5);
            *(_QWORD *)(v16 - 32) = *(v7 - 4);
          }
          while ( v7 != (_QWORD *)v17 );
        }
      }
      else
      {
        if ( v9 > *(unsigned int *)(a1 + 12) )
        {
          *(_DWORD *)(a1 + 8) = 0;
          v12 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, v9, 0x28u, &v24, a6);
          sub_2BF6E30(a1, v12);
          v22 = v24;
          if ( a1 + 16 != *(_QWORD *)a1 )
          {
            v23 = v24;
            _libc_free(*(_QWORD *)a1);
            v22 = v23;
          }
          *(_DWORD *)(a1 + 12) = v22;
          *(_QWORD *)a1 = v12;
          v7 = *(_QWORD **)a2;
          v9 = *(unsigned int *)(a2 + 8);
          v13 = *(_QWORD **)a2;
        }
        else
        {
          v12 = *(_QWORD **)a1;
          v13 = (_QWORD *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v19 = 40 * v10;
            v20 = a2 + 40 * v10 + 16;
            do
            {
              v21 = v7[4];
              v7 += 5;
              v12 += 5;
              *(v12 - 1) = v21;
              *(v12 - 3) = *(v7 - 3);
              *(v12 - 2) = *(v7 - 2);
              *(v12 - 5) = *(v7 - 5);
              *(v12 - 4) = *(v7 - 4);
            }
            while ( v7 != (_QWORD *)v20 );
            v7 = *(_QWORD **)a2;
            v9 = *(unsigned int *)(a2 + 8);
            v12 = (_QWORD *)(v19 + *(_QWORD *)a1);
            v13 = (_QWORD *)(*(_QWORD *)a2 + v19);
          }
        }
        for ( i = &v7[5 * v9]; i != v13; v12 += 5 )
        {
          if ( v12 )
          {
            *v12 = *v13;
            v12[1] = v13[1];
            v12[2] = v13[2];
            v12[3] = v13[3];
            v12[4] = v13[4];
          }
          v13 += 5;
        }
      }
      *(_DWORD *)(a1 + 8) = v11;
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v15 = *(_QWORD *)a1;
      if ( v15 != a1 + 16 )
      {
        _libc_free(v15);
        v8 = *(_QWORD **)a2;
      }
      *(_QWORD *)a1 = v8;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v7;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
