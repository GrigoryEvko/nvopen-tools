// Function: sub_FFF9B0
// Address: 0xfff9b0
//
unsigned __int64 __fastcall sub_FFF9B0(
        __int64 a1,
        unsigned __int8 *a2,
        int a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 *a6)
{
  unsigned int v7; // r14d
  __int64 v10; // rdx
  unsigned __int64 result; // rax
  char v12; // dl
  char v13; // r11
  unsigned __int8 *v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // r11
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // r12
  __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-40h]
  const __m128i *v22; // [rsp+8h] [rbp-38h]

  v7 = a5;
  v22 = (const __m128i *)a4;
  v10 = *(unsigned __int8 *)(a1 + 28);
  while ( 1 )
  {
    if ( !(_BYTE)v10 )
      goto LABEL_9;
    result = *(_QWORD *)(a1 + 8);
    a4 = *(unsigned int *)(a1 + 20);
    v10 = result + 8 * a4;
    if ( result != v10 )
      break;
LABEL_8:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a1 + 20) = a4;
      *(_QWORD *)v10 = a2;
      v10 = *(unsigned __int8 *)(a1 + 28);
      ++*(_QWORD *)a1;
      goto LABEL_10;
    }
LABEL_9:
    result = (unsigned __int64)sub_C8CC70(a1, (__int64)a2, v10, a4, a5, (__int64)a6);
    v13 = v12;
    v10 = *(unsigned __int8 *)(a1 + 28);
    if ( !v13 )
      return result;
LABEL_10:
    if ( ++v7 > 1 )
      return result;
    result = *a2;
    if ( (unsigned __int8)result <= 0x1Cu )
      return result;
    if ( a3 )
    {
      result = (unsigned int)(result - 48);
      switch ( (int)result )
      {
        case 0:
        case 3:
        case 7:
          if ( (a2[7] & 0x40) != 0 )
            a6 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            a6 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          a2 = *(unsigned __int8 **)a6;
          continue;
        case 9:
          if ( (a2[7] & 0x40) != 0 )
            v14 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            v14 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          sub_FFF9B0(a1, *(_QWORD *)v14, 1, v22, v7);
          if ( (a2[7] & 0x40) != 0 )
            a6 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
          else
            a6 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
          a2 = (unsigned __int8 *)*((_QWORD *)a6 + 4);
          v10 = *(unsigned __int8 *)(a1 + 28);
          continue;
        case 37:
          result = *((_QWORD *)a2 - 4);
          if ( result )
          {
            if ( !*(_BYTE *)result
              && *(_QWORD *)(result + 24) == *((_QWORD *)a2 + 10)
              && *(_DWORD *)(result + 36) == 371 )
            {
              result = -32LL * (*((_DWORD *)a2 + 1) & 0x7FFFFFF);
              a2 = *(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
              if ( a2 )
                continue;
            }
          }
          break;
        default:
          return result;
      }
      return result;
    }
    if ( (_BYTE)result == 58 )
    {
      v15 = *((_QWORD *)a2 - 8);
      if ( !v15 )
        return result;
      v16 = *((_QWORD *)a2 - 4);
      if ( !v16 )
        return result;
    }
    else
    {
      if ( (_BYTE)result != 85 )
      {
        if ( (unsigned __int8)result > 0x36u )
          return result;
        v20 = 0x40540000000000LL;
        if ( !_bittest64(&v20, result) )
          return result;
        goto LABEL_54;
      }
      result = *((_QWORD *)a2 - 4);
      if ( !result )
        return result;
      if ( *(_BYTE *)result )
        return result;
      if ( *(_QWORD *)(result + 24) != *((_QWORD *)a2 + 10) )
        return result;
      if ( *(_DWORD *)(result + 36) != 359 )
        return result;
      result = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
      v15 = *(_QWORD *)&a2[-32 * result];
      if ( !v15 )
        return result;
      result = 32 * (1 - result);
      v16 = *(_QWORD *)&a2[result];
      if ( !v16 )
        return result;
    }
    v21 = v16;
    sub_FFF9B0(a1, v15, 0, v22, v7);
    sub_FFF9B0(a1, v21, 0, v22, v7);
    result = *a2;
    if ( (unsigned __int8)result > 0x36u )
      return result;
    v17 = 0x40540000000000LL;
    if ( !_bittest64(&v17, result) )
      return result;
    if ( (unsigned __int8)result <= 0x1Cu )
    {
      result = *((unsigned __int16 *)a2 + 1);
      goto LABEL_44;
    }
LABEL_54:
    result = (unsigned int)(result - 29);
LABEL_44:
    if ( (_DWORD)result != 17 )
      return result;
    if ( (a2[1] & 2) == 0 )
      return result;
    v18 = *((_QWORD *)a2 - 8);
    if ( !v18 )
      return result;
    v19 = *((_QWORD *)a2 - 4);
    if ( !v19 )
      return result;
    if ( (unsigned __int8)sub_9B6260(v18, v22, 0) )
      sub_FFF9B0(a1, v19, 0, v22, v7);
    result = sub_9B6260(v19, v22, 0);
    if ( !(_BYTE)result )
      return result;
    v10 = *(unsigned __int8 *)(a1 + 28);
    a2 = (unsigned __int8 *)v18;
  }
  while ( a2 != *(unsigned __int8 **)result )
  {
    result += 8LL;
    if ( v10 == result )
      goto LABEL_8;
  }
  return result;
}
