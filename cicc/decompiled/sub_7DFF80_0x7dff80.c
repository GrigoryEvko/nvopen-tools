// Function: sub_7DFF80
// Address: 0x7dff80
//
__int64 __fastcall sub_7DFF80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 result; // rax
  char v9; // dl
  const __m128i *v10; // rsi
  __int8 v11; // dl
  __int64 v12; // r14
  unsigned __int8 v13; // si
  __int64 v14; // r8
  __int64 v15; // rdx
  int v16; // ecx
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned __int8 v19; // dl
  __int64 v20; // r8
  char v21; // bl

  v7 = *(_QWORD *)(a1 + 72);
  result = *(unsigned __int8 *)(a1 + 56);
  v9 = *(_BYTE *)(v7 + 24);
  if ( v9 != 1 )
  {
    if ( (_BYTE)result == 73 && (*(_BYTE *)(a1 + 25) & 1) == 0 && v9 == 3 )
    {
      v10 = *(const __m128i **)(v7 + 16);
      v11 = v10[1].m128i_i8[8];
      result = (__int64)v10;
      if ( v11 == 1 )
      {
        while ( *(_BYTE *)(result + 56) == 91 )
        {
          result = *(_QWORD *)(*(_QWORD *)(result + 72) + 16LL);
          v11 = *(_BYTE *)(result + 24);
          if ( v11 != 1 )
            goto LABEL_36;
        }
      }
      else
      {
LABEL_36:
        if ( v11 == 3 )
        {
          result = *(_QWORD *)(result + 56);
          if ( *(_QWORD *)(v7 + 56) == result )
            return sub_730620(a1, v10);
        }
      }
    }
    return result;
  }
  v12 = *(_QWORD *)(v7 + 72);
  if ( !(_BYTE)result )
  {
    result = *(unsigned __int8 *)(v7 + 56);
    if ( (_BYTE)result != 3 )
    {
      if ( (_BYTE)result == 92 )
      {
        v20 = *(_QWORD *)(v7 + 72);
        v21 = *(_BYTE *)(v7 + 60);
        *(_QWORD *)(v7 + 16) = *(_QWORD *)(a1 + 16);
        sub_73D8E0(v7, 0x32u, *(_QWORD *)a1, *(_BYTE *)(a1 + 25) & 1, v20);
        v10 = (const __m128i *)v7;
        *(_BYTE *)(v7 + 60) = v21 & 3 | *(_BYTE *)(v7 + 60) & 0xFC;
      }
      else
      {
        if ( (_BYTE)result != 8 || *(_BYTE *)(v12 + 24) != 1 || *(_BYTE *)(v12 + 56) != 3 )
          return result;
        v10 = (const __m128i *)sub_73E110(*(_QWORD *)(v12 + 72), *(_QWORD *)a1);
      }
      return sub_730620(a1, v10);
    }
    goto LABEL_34;
  }
  if ( (_BYTE)result != 3 )
  {
    if ( (_BYTE)result == 95 )
    {
      if ( *(_BYTE *)(v7 + 56) )
        return result;
      v13 = 94;
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v7 + 16);
      v14 = *(_QWORD *)(v7 + 72);
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE *)(a1 + 25) & 1;
    }
    else
    {
      if ( (_BYTE)result != 94 || *(_BYTE *)(v7 + 56) != 3 )
        return result;
      v13 = 95;
      *(_QWORD *)(v12 + 16) = *(_QWORD *)(v7 + 16);
      v14 = *(_QWORD *)(v7 + 72);
      v15 = *(_QWORD *)a1;
      v16 = *(_BYTE *)(a1 + 25) & 1;
    }
    return sub_73D8E0(a1, v13, v15, v16, v14);
  }
  if ( !*(_BYTE *)(v7 + 56) )
  {
    v17 = *(_QWORD *)a1;
    v18 = *(_QWORD *)v12;
    if ( v17 == *(_QWORD *)v12 || (result = sub_8D97D0(v17, v18, 1, a4, a5), (_DWORD)result) )
    {
      result = *(unsigned __int8 *)(a1 + 25);
      v19 = *(_BYTE *)(v12 + 25);
      if ( ((v19 ^ *(_BYTE *)(a1 + 25)) & 1) != 0 )
      {
        if ( (result & 1) != 0 || (*(_BYTE *)(v12 + 25) & 1) == 0 )
          return result;
        v10 = (const __m128i *)sub_731370(v12, v18, *(_BYTE *)(v12 + 25) & 1, (v19 ^ *(_BYTE *)(a1 + 25)) & 1, a5, a6);
        return sub_730620(a1, v10);
      }
LABEL_34:
      v10 = (const __m128i *)v12;
      return sub_730620(a1, v10);
    }
  }
  return result;
}
