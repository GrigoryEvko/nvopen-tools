// Function: sub_1EE7EC0
// Address: 0x1ee7ec0
//
__int64 __fastcall sub_1EE7EC0(
        __int64 a1,
        __int64 a2,
        unsigned __int16 *a3,
        unsigned __int16 *a4,
        __int64 a5,
        int a6,
        __int64 a7)
{
  unsigned __int16 *v8; // r13
  unsigned int v9; // ebx
  __int64 result; // rax
  __int64 v11; // rdx
  int v12; // edi
  unsigned int v13; // r11d
  unsigned int v14; // esi
  unsigned int v15; // edx
  unsigned int v16; // r15d
  unsigned __int16 v17; // r9
  __int64 v18; // rdi
  signed int v19; // r12d
  __int64 v20; // r10
  int v21; // eax
  unsigned __int16 *v22; // rdx
  unsigned __int16 *v24; // [rsp+10h] [rbp-60h]
  _DWORD *v25; // [rsp+18h] [rbp-58h]
  unsigned __int16 v27; // [rsp+2Ch] [rbp-44h]
  unsigned __int16 *v28; // [rsp+38h] [rbp-38h]

  v8 = a3;
  v9 = 0;
  result = (__int64)(a3 + 32);
  v28 = a3 + 32;
  do
  {
    v17 = *v8;
    if ( !*v8 )
      break;
    v18 = *(_QWORD *)(a1 + 16);
    v19 = v17 - 1;
    v20 = 4LL * v19;
    result = *(unsigned int *)(v20 + *(_QWORD *)(v18 + 88));
    if ( !(_DWORD)result )
    {
      v24 = a4;
      v25 = (_DWORD *)(v20 + *(_QWORD *)(v18 + 88));
      v27 = *v8;
      v21 = sub_1ED7BB0(v18, v19);
      a4 = v24;
      v20 = 4LL * v19;
      *v25 = v21;
      v17 = v27;
      result = *(unsigned int *)(*(_QWORD *)(v18 + 88) + v20);
    }
    v11 = *(_QWORD *)(a1 + 264);
    if ( v11 != *(_QWORD *)(a1 + 272) )
      result = (unsigned int)(*(_DWORD *)(v11 + 4LL * v19) + result);
    v12 = (__int16)v8[1];
    v13 = *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * v19);
    v14 = *(_DWORD *)(**(_QWORD **)(a1 + 48) + 4LL * v19);
    v15 = v12 + v13;
    v16 = v12 + v13;
    if ( v14 >= v12 + v13 )
      v16 = *(_DWORD *)(**(_QWORD **)(a1 + 48) + 4LL * v19);
    if ( *a4 )
      goto LABEL_11;
    if ( (unsigned int)result >= v15 )
    {
      if ( (unsigned int)result >= v13 )
        goto LABEL_11;
      result = (unsigned int)result - v13;
      v12 = result;
LABEL_22:
      if ( !v12 )
        goto LABEL_11;
      goto LABEL_10;
    }
    if ( (unsigned int)result < v13 )
      goto LABEL_22;
    LOWORD(v12) = v15 - result;
LABEL_10:
    *a4 = v17;
    a4[1] = v12;
LABEL_11:
    if ( v14 < v15 )
    {
      if ( !a4[2] && a6 != v9 )
      {
        while ( 1 )
        {
          v22 = (unsigned __int16 *)(a5 + 4LL * v9);
          result = (unsigned int)*v22 - 1;
          if ( v19 <= (unsigned int)result )
            break;
          if ( a6 == ++v9 )
            goto LABEL_13;
        }
        if ( v19 == (_DWORD)result )
        {
          result = v16 - (__int16)v22[1];
          if ( (unsigned int)(result - 1) <= 0x7FFE )
          {
            a4[2] = v17;
            a4[3] = result;
          }
        }
      }
LABEL_13:
      if ( !a4[4] )
      {
        result = a7;
        if ( *(_DWORD *)(v20 + a7) < v16 )
        {
          a4[4] = v17;
          a4[5] = v16 - v14;
        }
      }
    }
    v8 += 2;
  }
  while ( v8 != v28 );
  return result;
}
