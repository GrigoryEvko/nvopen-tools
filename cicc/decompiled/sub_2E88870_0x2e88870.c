// Function: sub_2E88870
// Address: 0x2e88870
//
void __fastcall sub_2E88870(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  int *v5; // rcx
  int v6; // eax
  unsigned __int64 v7; // rdx
  __int64 v8; // rax
  int *v9; // rcx
  int v10; // eax
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  unsigned __int8 *v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rax
  unsigned __int8 *v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned __int8 *v19; // rdx
  __int64 v20; // r8

  if ( a1 != a3 )
  {
    v4 = *(_QWORD *)(a3 + 48);
    v5 = (int *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v6 = v4 & 7;
      v7 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v6 != 1 )
      {
        v7 = 0;
        if ( v6 == 3 )
        {
          if ( *((_BYTE *)v5 + 4) )
            v7 = *(_QWORD *)&v5[2 * *v5 + 4];
        }
      }
    }
    else
    {
      v7 = 0;
    }
    sub_2E87C90(a1, a2, v7);
    v8 = *(_QWORD *)(a3 + 48);
    v9 = (int *)(v8 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v8 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v10 = v8 & 7;
      v11 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 != 2 )
      {
        v11 = 0;
        if ( v10 == 3 )
        {
          if ( *((_BYTE *)v9 + 5) )
            v11 = *(_QWORD *)&v9[2 * *((unsigned __int8 *)v9 + 4) + 4 + 2 * (__int64)*v9];
        }
      }
    }
    else
    {
      v11 = 0;
    }
    sub_2E87EC0(a1, a2, v11);
    v12 = *(_QWORD *)(a3 + 48);
    v13 = (unsigned __int8 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v14 = 0;
      if ( (v12 & 7) == 3 && v13[6] )
        v14 = *(_QWORD *)&v13[8 * *(int *)v13 + 16 + 8 * (__int64)(v13[5] + v13[4])];
    }
    else
    {
      v14 = 0;
    }
    sub_2E880E0(a1, a2, v14);
    v15 = *(_QWORD *)(a3 + 48);
    v16 = (unsigned __int8 *)(v15 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v15 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v17 = 0;
      if ( (v15 & 7) == 3 && v16[7] )
        v17 = *(_QWORD *)&v16[8 * v16[6] + 16 + 8 * *(int *)v16 + 8 * (__int64)(v16[5] + v16[4])];
    }
    else
    {
      v17 = 0;
    }
    sub_2E882B0(a1, a2, v17);
    v18 = *(_QWORD *)(a3 + 48);
    v19 = (unsigned __int8 *)(v18 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v20 = 0;
      if ( (v18 & 7) == 3 )
      {
        if ( v19[9] )
          v20 = *(_QWORD *)&v19[8 * v19[7] + 16 + 8 * v19[6] + 8 * *(int *)v19 + 8 * (__int64)(v19[5] + v19[4])];
      }
    }
    else
    {
      v20 = 0;
    }
    sub_2E88680(a1, a2, v20);
  }
}
