// Function: sub_1F21560
// Address: 0x1f21560
//
__int64 __fastcall sub_1F21560(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r10
  __int64 i; // r8
  __int64 v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdx
  __int64 *v12; // r8
  __int64 v13; // rcx
  __int64 v14; // rax

  result = a3 - 1;
  v6 = (a3 - 1) / 2;
  if ( a2 >= v6 )
  {
    v10 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v9 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v9 )
  {
    v9 = 2 * (i + 1);
    v10 = (__int64 *)(a1 + 16 * (i + 1));
    if ( (*(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v10 >> 1) & 3) < (*(_DWORD *)((*(v10 - 1) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                            | (unsigned int)(*(v10 - 1) >> 1)
                                                                                            & 3) )
    {
      --v9;
      v10 = (__int64 *)(a1 + 8 * v9);
    }
    *(_QWORD *)(a1 + 8 * i) = *v10;
    if ( v9 >= v6 )
      break;
  }
  if ( (a3 & 1) == 0 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v9 )
    {
      v13 = 2 * v9 + 2;
      v14 = *(_QWORD *)(a1 + 8 * v13 - 8);
      v9 = v13 - 1;
      *v10 = v14;
      v10 = (__int64 *)(a1 + 8 * v9);
    }
  }
  result = v9 - 1;
  v11 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 8 * v11);
      v10 = (__int64 *)(a1 + 8 * v9);
      result = *(_DWORD *)((*v12 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v12 >> 1) & 3;
      if ( (unsigned int)result >= ((unsigned int)(a4 >> 1) & 3 | *(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24)) )
        break;
      *v10 = *v12;
      v9 = v11;
      result = (v11 - 1) / 2;
      if ( a2 >= v11 )
      {
        *v12 = a4;
        return result;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = a4;
  return result;
}
