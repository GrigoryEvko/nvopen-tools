// Function: sub_2DCA840
// Address: 0x2dca840
//
__int64 __fastcall sub_2DCA840(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 i; // r15
  __int64 v7; // r12
  __int64 v8; // rcx
  unsigned int *v9; // r13
  unsigned int v10; // eax
  unsigned int *v11; // rcx
  __int64 v12; // r13
  unsigned int *v13; // r15
  unsigned int v14; // eax
  unsigned int v16; // eax
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h]
  unsigned int *v20; // [rsp+20h] [rbp-40h]
  unsigned int v22; // [rsp+2Ch] [rbp-34h]
  unsigned int v23; // [rsp+2Ch] [rbp-34h]

  v19 = (a3 - 1) / 2;
  v18 = a3 & 1;
  if ( a2 >= v19 )
  {
    v7 = a2;
    v11 = (unsigned int *)(a1 + 4 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v8 = 8 * (i + 1);
    v9 = (unsigned int *)(a1 + v8 - 4);
    v20 = (unsigned int *)(a1 + v8);
    v22 = sub_2DCA6E0(*(_QWORD *)(a5 + 8), *(unsigned int *)(a1 + v8));
    v10 = sub_2DCA6E0(*(_QWORD *)(a5 + 8), *v9);
    v11 = v20;
    if ( v22 > v10 )
    {
      --v7;
      v11 = (unsigned int *)(a1 + 4 * v7);
    }
    *(_DWORD *)(a1 + 4 * i) = *v11;
    if ( v7 >= v19 )
      break;
  }
  if ( !v18 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v7 )
    {
      v16 = *(_DWORD *)(a1 + 4 * (2 * v7 + 2) - 4);
      v7 = 2 * v7 + 1;
      *v11 = v16;
      v11 = (unsigned int *)(a1 + 4 * v7);
    }
  }
  v12 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v13 = (unsigned int *)(a1 + 4 * v12);
      v23 = sub_2DCA6E0(*(_QWORD *)(a5 + 8), *v13);
      v14 = sub_2DCA6E0(*(_QWORD *)(a5 + 8), a4);
      v11 = (unsigned int *)(a1 + 4 * v7);
      if ( v23 <= v14 )
        break;
      v7 = v12;
      *v11 = *v13;
      if ( a2 >= v12 )
      {
        v11 = (unsigned int *)(a1 + 4 * v12);
        break;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v11 = a4;
  return a4;
}
