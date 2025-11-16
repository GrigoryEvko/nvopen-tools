// Function: sub_9487F0
// Address: 0x9487f0
//
_BYTE *__fastcall sub_9487F0(__int64 a1, __int64 a2)
{
  _BYTE *v4; // rsi
  _BYTE *result; // rax
  _BYTE *v6; // rdi
  __int64 v7; // rdx
  size_t v8; // rdx
  _BYTE *v9; // rdi

  v4 = *(_BYTE **)a2;
  result = (_BYTE *)(a2 + 16);
  v6 = *(_BYTE **)a1;
  if ( v4 == (_BYTE *)(a2 + 16) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    if ( v8 )
    {
      if ( v8 == 1 )
      {
        result = (_BYTE *)*(unsigned __int8 *)(a2 + 16);
        *v6 = (_BYTE)result;
      }
      else
      {
        result = memcpy(v6, v4, v8);
      }
      v8 = *(_QWORD *)(a2 + 8);
      v6 = *(_BYTE **)a1;
    }
    *(_QWORD *)(a1 + 8) = v8;
    v6[v8] = 0;
    v9 = *(_BYTE **)a2;
    *(_QWORD *)(a2 + 8) = 0;
    *v9 = 0;
  }
  else
  {
    if ( v6 == (_BYTE *)(a1 + 16) )
    {
      *(_QWORD *)a1 = v4;
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
    }
    else
    {
      *(_QWORD *)a1 = v4;
      v7 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 8) = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 + 16) = *(_QWORD *)(a2 + 16);
      if ( v6 )
      {
        *(_QWORD *)a2 = v6;
        *(_QWORD *)(a2 + 16) = v7;
        *(_QWORD *)(a2 + 8) = 0;
        *v6 = 0;
        return result;
      }
    }
    *(_QWORD *)a2 = result;
    *(_QWORD *)(a2 + 8) = 0;
    *result = 0;
  }
  return result;
}
