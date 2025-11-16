// Function: sub_15B93D0
// Address: 0x15b93d0
//
__int64 __fastcall sub_15B93D0(__int64 a1, unsigned int **a2, unsigned int ***a3)
{
  int v4; // ebx
  unsigned int *v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rdx
  int v11; // eax
  unsigned int *v12; // rsi
  unsigned int **v13; // rdi
  unsigned int v14; // eax
  int v15; // r8d
  unsigned int **v16; // rcx
  unsigned int *v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h] BYREF
  unsigned int v20; // [rsp+10h] [rbp-50h] BYREF
  __int64 v21; // [rsp+18h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-38h] BYREF
  __int64 v24[6]; // [rsp+30h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = v4 - 1;
    v10 = (*a2)[2];
    v18 = *(_QWORD *)&v6[-2 * v6[2]];
    v19 = *(_QWORD *)&v6[2 * (1 - v10)];
    v20 = v6[6];
    v21 = *(_QWORD *)&v6[2 * (2 - v10)];
    v22 = *(_QWORD *)&v6[2 * (3 - v10)];
    v23 = v6[7];
    v24[0] = *(_QWORD *)&v6[2 * (4 - v10)];
    v11 = sub_15B52D0(&v18, &v19, (int *)&v20, &v21, &v22, (int *)&v23, v24);
    v12 = *a2;
    v13 = 0;
    v14 = v9 & v11;
    v15 = 1;
    v16 = (unsigned int **)(v7 + 8LL * v14);
    v17 = *v16;
    if ( *a2 == *v16 )
    {
LABEL_10:
      *a3 = v16;
      return 1;
    }
    else
    {
      while ( v17 != (unsigned int *)-8LL )
      {
        if ( v17 != (unsigned int *)-16LL || v13 )
          v16 = v13;
        v14 = v9 & (v15 + v14);
        v17 = *(unsigned int **)(v7 + 8LL * v14);
        if ( v17 == v12 )
        {
          v16 = (unsigned int **)(v7 + 8LL * v14);
          goto LABEL_10;
        }
        ++v15;
        v13 = v16;
        v16 = (unsigned int **)(v7 + 8LL * v14);
      }
      if ( !v13 )
        v13 = v16;
      *a3 = v13;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
