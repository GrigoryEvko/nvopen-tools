// Function: sub_2ECEBE0
// Address: 0x2ecebe0
//
__int64 __fastcall sub_2ECEBE0(__int64 a1, __int64 a2, unsigned int a3, char a4, unsigned int a5)
{
  __int64 v5; // r14
  __int64 result; // rax
  unsigned int v10; // esi
  unsigned int v11; // edi
  unsigned int v12; // ecx
  __int64 v13; // rcx
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  _QWORD v17[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = a5;
  result = *(unsigned int *)(a1 + 164);
  v10 = *(_DWORD *)(a1 + 172);
  if ( (unsigned int)result >= a3 )
  {
    if ( a3 >= v10 )
      goto LABEL_11;
  }
  else
  {
    v11 = *(_DWORD *)(a1 + 712);
    v12 = a3 - result;
    if ( a3 - (unsigned int)result < v11 )
      v12 = v11;
    *(_DWORD *)(a1 + 712) = v12;
    if ( a3 >= v10 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
        goto LABEL_11;
      goto LABEL_6;
    }
  }
  v13 = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 172) = a3;
  if ( *(_DWORD *)(v13 + 4) )
    goto LABEL_11;
LABEL_6:
  if ( (unsigned int)result < a3 )
    goto LABEL_7;
LABEL_11:
  result = sub_2ECEA00(a1, (_QWORD *)a2);
  if ( (_BYTE)result
    || (v14 = *(_BYTE **)(a1 + 72),
        result = (__int64)&v14[-*(_QWORD *)(a1 + 64)] >> 3,
        (unsigned int)qword_5021648 <= (unsigned int)result) )
  {
LABEL_7:
    if ( !a4 )
    {
      v17[0] = a2;
      v16 = *(_BYTE **)(a1 + 136);
      if ( v16 == *(_BYTE **)(a1 + 144) )
      {
        sub_2ECAD30(a1 + 128, v16, v17);
        a2 = v17[0];
      }
      else
      {
        if ( v16 )
        {
          *(_QWORD *)v16 = a2;
          v16 = *(_BYTE **)(a1 + 136);
        }
        *(_QWORD *)(a1 + 136) = v16 + 8;
      }
      result = *(unsigned int *)(a1 + 88);
      *(_DWORD *)(a2 + 204) |= result;
    }
    return result;
  }
  v17[0] = a2;
  if ( v14 == *(_BYTE **)(a1 + 80) )
  {
    sub_2ECAD30(a1 + 64, v14, v17);
    a2 = v17[0];
  }
  else
  {
    if ( v14 )
    {
      *(_QWORD *)v14 = a2;
      v14 = *(_BYTE **)(a1 + 72);
    }
    *(_QWORD *)(a1 + 72) = v14 + 8;
  }
  result = *(unsigned int *)(a1 + 24);
  *(_DWORD *)(a2 + 204) |= result;
  if ( a4 )
  {
    v15 = *(_QWORD *)(a1 + 128) + 8 * v5;
    *(_DWORD *)(*(_QWORD *)v15 + 204LL) &= ~*(_DWORD *)(a1 + 88);
    result = *(_QWORD *)(*(_QWORD *)(a1 + 136) - 8LL);
    *(_QWORD *)v15 = result;
    *(_QWORD *)(a1 + 136) -= 8LL;
  }
  return result;
}
