// Function: sub_386D540
// Address: 0x386d540
//
_QWORD *__fastcall sub_386D540(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r12
  _QWORD *v11; // rbx
  __int64 v12; // rax
  _QWORD *result; // rax
  __int64 v14; // rcx
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx

  v10 = *(_QWORD **)(a1 + 8);
  v11 = &v10[3 * *(unsigned int *)(a1 + 16)];
  while ( v10 != v11 )
  {
    while ( 1 )
    {
      v12 = *(v11 - 1);
      v11 -= 3;
      if ( v12 == 0 || v12 == -8 || v12 == -16 )
        break;
      sub_1649B30(v11);
      if ( v10 == v11 )
        goto LABEL_6;
    }
  }
LABEL_6:
  *(_DWORD *)(a1 + 16) = 0;
  result = sub_386D460((__int64 *)a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( *(_QWORD *)(a2 - 24) )
  {
    v14 = *(_QWORD *)(a2 - 16);
    v15 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *(_QWORD *)(a2 - 24) = result;
  if ( result )
  {
    v16 = result[1];
    *(_QWORD *)(a2 - 16) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (a2 - 16) | *(_QWORD *)(v16 + 16) & 3LL;
    *(_QWORD *)(a2 - 24 + 16) = (unsigned __int64)(result + 1) | *(_QWORD *)(a2 - 8) & 3LL;
    result[1] = a2 - 24;
  }
  return result;
}
