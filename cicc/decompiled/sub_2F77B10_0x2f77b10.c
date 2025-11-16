// Function: sub_2F77B10
// Address: 0x2f77b10
//
__int64 __fastcall sub_2F77B10(
        __int64 a1,
        __int64 a2,
        unsigned __int16 *a3,
        unsigned __int16 *a4,
        __int64 a5,
        int a6,
        __int64 a7)
{
  unsigned __int16 *v9; // r14
  unsigned int v10; // ebx
  __int64 result; // rax
  __int64 v12; // rdx
  int v13; // edi
  unsigned int v14; // r11d
  unsigned int v15; // esi
  unsigned int v16; // edx
  unsigned int v17; // r12d
  unsigned __int16 v18; // r9
  __int64 v19; // rdx
  signed int v20; // r13d
  int v21; // eax
  unsigned __int16 *v22; // rdx
  unsigned __int16 *v24; // [rsp+18h] [rbp-58h]
  int v25; // [rsp+20h] [rbp-50h]
  unsigned __int16 v26; // [rsp+24h] [rbp-4Ch]
  __int64 v27; // [rsp+28h] [rbp-48h]
  unsigned __int16 *v28; // [rsp+30h] [rbp-40h]

  v9 = a3;
  v10 = 0;
  result = (__int64)(a3 + 32);
  v28 = a3 + 32;
  do
  {
    v18 = *v9;
    if ( !*v9 )
      break;
    v19 = *(_QWORD *)(a1 + 16);
    v20 = v18 - 1;
    result = *(unsigned int *)(*(_QWORD *)(v19 + 296) + 4LL * v20);
    if ( !(_DWORD)result )
    {
      v24 = a4;
      v25 = a6;
      v26 = *v9;
      v27 = *(_QWORD *)(a1 + 16);
      v21 = sub_2F60A40(v19, v20);
      a4 = v24;
      a6 = v25;
      v18 = v26;
      *(_DWORD *)(*(_QWORD *)(v27 + 296) + 4LL * v20) = v21;
      result = *(unsigned int *)(*(_QWORD *)(v27 + 296) + 4LL * v20);
    }
    v12 = *(_QWORD *)(a1 + 392);
    if ( v12 != *(_QWORD *)(a1 + 400) )
      result = (unsigned int)(*(_DWORD *)(v12 + 4LL * v20) + result);
    v13 = (__int16)v9[1];
    v14 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * v20);
    v15 = *(_DWORD *)(**(_QWORD **)(a1 + 48) + 4LL * v20);
    v16 = v13 + v14;
    v17 = v13 + v14;
    if ( v15 >= v13 + v14 )
      v17 = *(_DWORD *)(**(_QWORD **)(a1 + 48) + 4LL * v20);
    if ( *a4 )
      goto LABEL_11;
    if ( (unsigned int)result >= v16 )
    {
      if ( (unsigned int)result >= v14 )
        goto LABEL_11;
      result = (unsigned int)result - v14;
      v13 = result;
LABEL_22:
      if ( !v13 )
        goto LABEL_11;
      goto LABEL_10;
    }
    if ( (unsigned int)result < v14 )
      goto LABEL_22;
    LOWORD(v13) = v16 - result;
LABEL_10:
    *a4 = v18;
    a4[1] = v13;
LABEL_11:
    if ( v15 < v16 )
    {
      if ( !a4[2] && a6 != v10 )
      {
        while ( 1 )
        {
          v22 = (unsigned __int16 *)(a5 + 4LL * v10);
          result = (unsigned int)*v22 - 1;
          if ( v20 <= (unsigned int)result )
            break;
          if ( a6 == ++v10 )
            goto LABEL_13;
        }
        if ( v20 == (_DWORD)result )
        {
          result = v17 - (__int16)v22[1];
          if ( (unsigned int)(result - 1) <= 0x7FFE )
          {
            a4[2] = v18;
            a4[3] = result;
          }
        }
      }
LABEL_13:
      if ( !a4[4] )
      {
        result = 4LL * v20;
        if ( *(_DWORD *)(result + a7) < v17 )
        {
          a4[4] = v18;
          a4[5] = v17 - v15;
        }
      }
    }
    v9 += 2;
  }
  while ( v9 != v28 );
  return result;
}
