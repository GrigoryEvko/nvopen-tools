// Function: sub_C653C0
// Address: 0xc653c0
//
__int64 __fastcall sub_C653C0(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  __int64 result; // rax
  bool v6; // zf
  __int64 v7; // rdx
  __int64 v8; // r8
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // r8
  __int64 v11; // r8
  __int64 v12; // rdx
  unsigned int v13; // r15d
  int v14; // ebx
  int v15; // edx
  int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rbx
  size_t v19; // r8
  int v20; // [rsp+4h] [rbp-3Ch]
  unsigned int v21; // [rsp+8h] [rbp-38h]
  unsigned int v22; // [rsp+8h] [rbp-38h]

  result = a3;
  v6 = a3 == 0;
  v7 = *(unsigned int *)(a1 + 8);
  v8 = !v6 + ((a3 - !v6) >> 2) + 1;
  v9 = *(unsigned int *)(a1 + 12);
  v10 = v7 + v8;
  if ( v10 > v9 )
  {
    sub_C8D5F0(a1, a1 + 16, v10, 4);
    v7 = *(unsigned int *)(a1 + 8);
    result = a3;
    v11 = v7 + 1;
    if ( v7 + 1 <= (unsigned __int64)*(unsigned int *)(a1 + 12) )
      goto LABEL_3;
  }
  else
  {
    v11 = v7 + 1;
    if ( v7 + 1 <= v9 )
      goto LABEL_3;
  }
  v21 = result;
  sub_C8D5F0(a1, a1 + 16, v11, 4);
  v7 = *(unsigned int *)(a1 + 8);
  result = v21;
LABEL_3:
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v7) = a3;
  v12 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
  *(_DWORD *)(a1 + 8) = v12;
  if ( !a3 )
    return result;
  if ( ((unsigned __int8)a2 & 3) == 0 )
  {
    v18 = a3 >> 2;
    v19 = 4 * v18;
    if ( v18 + v12 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, a1 + 16, v18 + v12, 4);
      v12 = *(unsigned int *)(a1 + 8);
      v19 = 4 * v18;
    }
    if ( v19 )
    {
      memcpy((void *)(*(_QWORD *)a1 + 4 * v12), a2, v19);
      LODWORD(v12) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v12 + v18;
    result = 4 * (a3 >> 2) + 4 - a3;
    if ( (_DWORD)result != 2 )
      goto LABEL_10;
LABEL_23:
    v15 = 0;
    goto LABEL_13;
  }
  v13 = 4;
  if ( a3 > 3 )
  {
    do
    {
      v14 = (a2[v13 - 3] << 8) | a2[v13 - 4] | (a2[v13 - 2] << 16) | (a2[v13 - 1] << 24);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v20 = result;
        sub_C8D5F0(a1, a1 + 16, v12 + 1, 4);
        v12 = *(unsigned int *)(a1 + 8);
        LODWORD(result) = v20;
      }
      v13 += 4;
      *(_DWORD *)(*(_QWORD *)a1 + 4 * v12) = v14;
      v12 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
      *(_DWORD *)(a1 + 8) = v12;
    }
    while ( (unsigned int)result >= v13 );
  }
  result = v13 - a3;
  if ( (_DWORD)result == 2 )
    goto LABEL_23;
LABEL_10:
  if ( (_DWORD)result == 3 )
  {
    v16 = 0;
    goto LABEL_14;
  }
  if ( (_DWORD)result != 1 )
    return result;
  v15 = a2[a3 - 3] << 8;
LABEL_13:
  v16 = (v15 | a2[a3 - 2]) << 8;
LABEL_14:
  result = a2[a3 - 1] | (unsigned int)v16;
  v17 = *(unsigned int *)(a1 + 8);
  if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    v22 = result;
    sub_C8D5F0(a1, a1 + 16, v17 + 1, 4);
    v17 = *(unsigned int *)(a1 + 8);
    result = v22;
  }
  *(_DWORD *)(*(_QWORD *)a1 + 4 * v17) = result;
  ++*(_DWORD *)(a1 + 8);
  return result;
}
