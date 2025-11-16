// Function: sub_893800
// Address: 0x893800
//
__int64 **__fastcall sub_893800(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v6; // rdi
  __int64 **v7; // rax
  __int64 v8; // rbx
  __int64 *v9; // rcx
  _QWORD *v10; // rsi
  __int64 **v11; // rdx
  _QWORD *v12; // rax
  __int64 **result; // rax
  __int64 *v14; // r8
  __int64 v15; // r14
  __m128i *v16; // rax

  v6 = (unsigned __int64)qword_4D03B88;
  v7 = qword_4D03B88;
  if ( qword_4D03B88 )
  {
    do
    {
      v7[5] = *(__int64 **)(a1 + 192);
      v7 = (__int64 **)*v7;
    }
    while ( v7 );
    v8 = *(_QWORD *)(a3 + 400);
    if ( !v8 )
    {
      v9 = *(__int64 **)(a3 + 288);
      goto LABEL_6;
    }
    goto LABEL_4;
  }
  v8 = *(_QWORD *)(a3 + 400);
  if ( v8 )
  {
LABEL_4:
    switch ( *(_BYTE *)(v8 + 80) )
    {
      case 4:
      case 5:
        v15 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 80LL);
        break;
      case 6:
        v15 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v15 = *(_QWORD *)(*(_QWORD *)(v8 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v15 = *(_QWORD *)(v8 + 88);
        break;
      default:
        sub_679050((_QWORD ***)v6);
        BUG();
    }
    sub_679050((_QWORD ***)v6);
    v16 = sub_678F80(*(const __m128i **)(v15 + 288));
    v6 = (unsigned __int64)v16;
    if ( v16 )
    {
      do
      {
        v16[2].m128i_i64[1] = *(_QWORD *)(a1 + 192);
        v16 = (__m128i *)v16->m128i_i64[0];
      }
      while ( v16 );
      v9 = *(__int64 **)(a3 + 288);
      qword_4D03B88 = (__int64 **)v6;
    }
    else
    {
      result = (__int64 **)&qword_4D03B88;
      v9 = *(__int64 **)(a3 + 288);
      qword_4D03B88 = 0;
      if ( !v9 )
        return result;
    }
    goto LABEL_6;
  }
  v9 = *(__int64 **)(a3 + 288);
  if ( !v9 )
  {
LABEL_26:
    result = (__int64 **)&dword_4D047B0;
    if ( dword_4D047B0 )
    {
      if ( !*(_QWORD *)(a1 + 240) || (result = (__int64 **)*(unsigned int *)(a1 + 100), (_DWORD)result) )
      {
        *(_DWORD *)(*(_QWORD *)(a1 + 192) + 44LL) = ++dword_4F066AC;
        return sub_893600(a2, qword_4D03B88, *(_QWORD *)(a1 + 440), 1);
      }
    }
    return result;
  }
LABEL_6:
  v10 = 0;
  v11 = 0;
  while ( 1 )
  {
    if ( v6 )
    {
      if ( v9 )
      {
        v12 = (_QWORD *)v9[7];
        if ( *(_QWORD *)(v6 + 56) >= (unsigned __int64)v12 )
        {
          if ( *(_QWORD **)(v6 + 56) == v12 )
            v6 = *(_QWORD *)v6;
          result = (__int64 **)v9;
          v9 = (__int64 *)*v9;
          v14 = (__int64 *)(v6 | (unsigned __int64)v9);
        }
        else
        {
          result = (__int64 **)v6;
          v6 = *(_QWORD *)v6;
          v14 = (__int64 *)((unsigned __int64)v9 | v6);
        }
      }
      else
      {
        v14 = *(__int64 **)v6;
        result = (__int64 **)v6;
        v6 = *(_QWORD *)v6;
      }
    }
    else
    {
      v14 = (__int64 *)*v9;
      result = (__int64 **)v9;
      v9 = (__int64 *)*v9;
    }
    if ( !v11 )
      v11 = result;
    if ( v10 )
      *v10 = result;
    *result = 0;
    if ( !v14 )
      break;
    v10 = result;
  }
  *(_QWORD *)(a3 + 288) = v11;
  if ( !v8 )
    goto LABEL_26;
  return result;
}
