// Function: sub_15C5B60
// Address: 0x15c5b60
//
unsigned int *__fastcall sub_15C5B60(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        unsigned int a9,
        char a10)
{
  __int64 v10; // r11
  __int64 v14; // r10
  unsigned int *result; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r14
  int v20; // eax
  unsigned int v21; // ecx
  unsigned int **v22; // rax
  unsigned int *v23; // rdx
  __int64 v24; // rsi
  int i; // [rsp+1Ch] [rbp-A4h]
  __int64 v26; // [rsp+30h] [rbp-90h]
  int v27; // [rsp+38h] [rbp-88h]
  __int64 v28; // [rsp+40h] [rbp-80h]
  __int64 v30; // [rsp+50h] [rbp-70h] BYREF
  __int64 v31; // [rsp+58h] [rbp-68h] BYREF
  __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  __int64 v33; // [rsp+68h] [rbp-58h] BYREF
  __int64 v34; // [rsp+70h] [rbp-50h] BYREF
  int v35; // [rsp+78h] [rbp-48h] BYREF
  __int64 v36[8]; // [rsp+80h] [rbp-40h] BYREF

  v10 = a2;
  if ( a9 )
  {
LABEL_4:
    v32 = a5;
    v33 = a6;
    v34 = a8;
    v16 = *a1;
    v30 = v10;
    v31 = a3;
    v17 = v16 + 1200;
    v18 = sub_161E980(32, 5);
    v19 = v18;
    if ( v18 )
    {
      sub_1623D80(v18, (_DWORD)a1, 27, a9, (unsigned int)&v30, 5, 0, 0);
      *(_WORD *)(v19 + 2) = 16896;
      *(_DWORD *)(v19 + 24) = a4;
      *(_DWORD *)(v19 + 28) = a7;
    }
    return sub_15C5980((unsigned int *)v19, a9, v17);
  }
  v14 = *a1;
  v30 = a2;
  v31 = a3;
  LODWORD(v32) = a4;
  v33 = a5;
  v35 = a7;
  v34 = a6;
  v36[0] = a8;
  v26 = v14;
  v28 = *(_QWORD *)(v14 + 1208);
  v27 = *(_DWORD *)(v14 + 1224);
  if ( !v27 )
    goto LABEL_3;
  v20 = sub_15B52D0(&v30, &v31, (int *)&v32, &v33, &v34, &v35, v36);
  v10 = a2;
  v21 = (v27 - 1) & v20;
  v22 = (unsigned int **)(v28 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == (unsigned int *)-8LL )
    goto LABEL_3;
  for ( i = 1; ; ++i )
  {
    if ( v23 != (unsigned int *)-16LL )
    {
      v24 = v23[2];
      if ( v30 == *(_QWORD *)&v23[-2 * v24]
        && v31 == *(_QWORD *)&v23[2 * (1 - v24)]
        && (_DWORD)v32 == v23[6]
        && v33 == *(_QWORD *)&v23[2 * (2 - v24)]
        && v34 == *(_QWORD *)&v23[2 * (3 - v24)]
        && v35 == v23[7]
        && v36[0] == *(_QWORD *)&v23[2 * (4 - v24)] )
      {
        break;
      }
    }
    v21 = (v27 - 1) & (i + v21);
    v22 = (unsigned int **)(v28 + 8LL * v21);
    v23 = *v22;
    if ( *v22 == (unsigned int *)-8LL )
      goto LABEL_3;
  }
  if ( v22 == (unsigned int **)(*(_QWORD *)(v26 + 1208) + 8LL * *(unsigned int *)(v26 + 1224)) || (result = *v22) == 0 )
  {
LABEL_3:
    result = 0;
    if ( !a10 )
      return result;
    goto LABEL_4;
  }
  return result;
}
