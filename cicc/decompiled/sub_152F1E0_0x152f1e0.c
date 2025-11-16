// Function: sub_152F1E0
// Address: 0x152f1e0
//
void __fastcall sub_152F1E0(_DWORD **a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v7; // rax
  _BOOL8 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *i; // r15
  int v15; // ecx
  __int64 v16; // r8
  int v17; // ecx
  _DWORD *v18; // r8
  unsigned int v19; // esi
  _DWORD *v20; // rdx
  __int64 v21; // r10
  int v22; // edx
  int v23; // r9d
  __int64 v24; // [rsp+8h] [rbp-38h]

  if ( !*a4 )
    *a4 = sub_1527880((_QWORD **)a1);
  v7 = *(unsigned int *)(a3 + 8);
  v8 = *(_BYTE *)(a2 + 1) == 1;
  if ( (unsigned int)v7 >= *(_DWORD *)(a3 + 12) )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  v9 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v9;
  v10 = *(unsigned __int16 *)(a2 + 2);
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v9 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v9 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  v11 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v11;
  if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v11 )
  {
    sub_16CD150(a3, a3 + 16, 0, 8);
    v11 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v11) = 0;
  v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  *(_DWORD *)(a3 + 8) = v12;
  v13 = 8LL * *(unsigned int *)(a2 + 8);
  for ( i = (_QWORD *)(a2 - v13); (_QWORD *)a2 != i; *(_DWORD *)(a3 + 8) = v12 )
  {
    v15 = *((_DWORD *)a1 + 76);
    v16 = 0;
    if ( v15 )
    {
      v17 = v15 - 1;
      v18 = a1[36];
      v19 = v17 & (((unsigned int)*i >> 9) ^ ((unsigned int)*i >> 4));
      v20 = &v18[4 * v19];
      v21 = *(_QWORD *)v20;
      if ( *i == *(_QWORD *)v20 )
      {
LABEL_12:
        v16 = (unsigned int)v20[3];
      }
      else
      {
        v22 = 1;
        while ( v21 != -4 )
        {
          v23 = v22 + 1;
          v19 = v17 & (v22 + v19);
          v20 = &v18[4 * v19];
          v21 = *(_QWORD *)v20;
          if ( *i == *(_QWORD *)v20 )
            goto LABEL_12;
          v22 = v23;
        }
        v16 = 0;
      }
    }
    if ( *(_DWORD *)(a3 + 12) <= (unsigned int)v12 )
    {
      v24 = v16;
      sub_16CD150(a3, a3 + 16, 0, 8);
      v12 = *(unsigned int *)(a3 + 8);
      v16 = v24;
    }
    ++i;
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v12) = v16;
    v12 = (unsigned int)(*(_DWORD *)(a3 + 8) + 1);
  }
  sub_152B6B0(*a1, 0xCu, a3, *a4);
  *(_DWORD *)(a3 + 8) = 0;
}
