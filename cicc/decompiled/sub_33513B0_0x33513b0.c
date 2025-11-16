// Function: sub_33513B0
// Address: 0x33513b0
//
__int64 __fastcall sub_33513B0(__int64 a1, unsigned int *a2, unsigned int *a3, __int64 a4)
{
  int v6; // eax
  unsigned int *v7; // rax
  unsigned int *v8; // rsi
  __int64 v9; // r12
  _QWORD *v11; // r9
  unsigned int v12; // eax
  _QWORD *v13; // r14
  unsigned int v14; // edx
  __int64 v15; // rax
  int v16; // eax
  unsigned int v18; // eax
  _QWORD *v19; // [rsp+8h] [rbp-58h]
  unsigned int v20; // [rsp+14h] [rbp-4Ch]
  __int64 v21; // [rsp+18h] [rbp-48h]
  unsigned int v22; // [rsp+28h] [rbp-38h] BYREF
  unsigned int v23[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = *(_DWORD *)(a1 + 24);
  while ( v6 != 2 )
  {
    if ( v6 < 0 )
    {
      v16 = ~v6;
      if ( *(_DWORD *)(a4 + 68) == v16 )
      {
        v18 = *a2 + 1;
        *a2 = v18;
        if ( *a3 >= v18 )
          v18 = *a3;
        *a3 = v18;
      }
      else if ( v16 == *(_DWORD *)(a4 + 64) && (*a2)-- == 1 )
      {
        return a1;
      }
    }
    v7 = *(unsigned int **)(a1 + 40);
    v8 = &v7[10 * *(unsigned int *)(a1 + 64)];
    if ( v7 == v8 )
      return 0;
    while ( 1 )
    {
      a1 = *(_QWORD *)v7;
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v7 + 48LL) + 16LL * v7[2]) == 1 )
        break;
      v7 += 10;
      if ( v8 == v7 )
        return 0;
    }
    v6 = *(_DWORD *)(a1 + 24);
    if ( v6 == 1 )
      return 0;
  }
  v11 = *(_QWORD **)(a1 + 40);
  v12 = *a3;
  v19 = &v11[5 * *(unsigned int *)(a1 + 64)];
  if ( v19 == v11 )
  {
    v20 = *a3;
    v9 = 0;
  }
  else
  {
    v20 = *a3;
    v13 = *(_QWORD **)(a1 + 40);
    v9 = 0;
    while ( 1 )
    {
      v14 = *a2;
      v23[0] = v12;
      v21 = a4;
      v22 = v14;
      v15 = sub_33513B0(*v13, &v22, v23);
      a4 = v21;
      if ( v15 && (!v9 || v20 < v23[0]) )
      {
        v20 = v23[0];
        v9 = v15;
      }
      v13 += 5;
      if ( v19 == v13 )
        break;
      v12 = *a3;
    }
  }
  *a3 = v20;
  return v9;
}
