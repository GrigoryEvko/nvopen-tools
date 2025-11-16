// Function: sub_239DC20
// Address: 0x239dc20
//
__int64 __fastcall sub_239DC20(
        __int64 *dest,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        __int64 a9)
{
  __int64 result; // rax
  unsigned __int64 v10; // rdx
  __int64 *v11; // r15
  __int64 *v12; // rbx
  signed __int64 v13; // r12
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 *i; // r14
  __int64 *src; // [rsp+8h] [rbp-38h]

  result = *dest;
  v10 = *dest & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*dest & 4) == 0 )
  {
    src = dest;
    v11 = dest;
    if ( !v10 )
      goto LABEL_3;
    src = dest + 1;
    goto LABEL_29;
  }
  v11 = *(__int64 **)v10;
  v14 = 8LL * *(unsigned int *)(v10 + 8);
  src = (__int64 *)(*(_QWORD *)v10 + v14);
  v15 = v14 >> 3;
  v16 = v14 >> 5;
  if ( v16 )
  {
    while ( !(unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
    {
      if ( (unsigned __int8)sub_BBD330(a7, v11[1], a8, a9) )
      {
        ++v11;
        goto LABEL_22;
      }
      if ( (unsigned __int8)sub_BBD330(a7, v11[2], a8, a9) )
      {
        v11 += 2;
        goto LABEL_22;
      }
      if ( (unsigned __int8)sub_BBD330(a7, v11[3], a8, a9) )
      {
        v11 += 3;
        goto LABEL_22;
      }
      v11 += 4;
      if ( !--v16 )
      {
        v15 = src - v11;
        goto LABEL_14;
      }
    }
    goto LABEL_22;
  }
LABEL_14:
  if ( v15 == 2 )
  {
LABEL_15:
    if ( (unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
      goto LABEL_22;
    ++v11;
    goto LABEL_29;
  }
  if ( v15 == 3 )
  {
    if ( (unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
      goto LABEL_22;
    ++v11;
    goto LABEL_15;
  }
  if ( v15 != 1 )
    goto LABEL_30;
LABEL_29:
  if ( !(unsigned __int8)sub_BBD330(a7, *v11, a8, a9) )
  {
LABEL_30:
    v11 = src;
    result = *dest;
    goto LABEL_3;
  }
LABEL_22:
  if ( v11 != src )
  {
    for ( i = v11 + 1; i != src; ++i )
    {
      if ( !(unsigned __int8)sub_BBD330(a7, *i, a8, a9) )
        *v11++ = *i;
    }
  }
  result = *dest;
LABEL_3:
  if ( ((result >> 2) & 1) != 0 )
  {
    if ( result )
    {
      if ( ((result >> 2) & 1) != 0 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v12 = (__int64 *)result;
        if ( result )
        {
          result = *(_QWORD *)result;
          v13 = result + 8LL * *((unsigned int *)v12 + 2) - (_QWORD)src;
          if ( src != (__int64 *)(result + 8LL * *((unsigned int *)v12 + 2)) )
          {
            memmove(v11, src, result + 8LL * *((unsigned int *)v12 + 2) - (_QWORD)src);
            result = *v12;
          }
          *((_DWORD *)v12 + 2) = ((__int64)v11 + v13 - result) >> 3;
        }
      }
    }
  }
  else if ( dest == v11 && v11 != src )
  {
    *dest = 0;
    return (__int64)dest;
  }
  return result;
}
