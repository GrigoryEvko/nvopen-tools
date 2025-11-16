// Function: sub_DF4EC0
// Address: 0xdf4ec0
//
__int64 __fastcall sub_DF4EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int8 **v9; // rbx
  __int64 v10; // rdx
  unsigned __int8 **v11; // r12
  unsigned __int8 *v12; // rsi
  unsigned __int8 v13; // al
  __int64 v14; // rax

  result = *(unsigned __int8 *)(a1 - 16);
  if ( (result & 2) != 0 )
  {
    v9 = *(unsigned __int8 ***)(a1 - 32);
    v10 = *(unsigned int *)(a1 - 24);
  }
  else
  {
    result = 8LL * (((unsigned __int8)result >> 2) & 0xF);
    v10 = (*(_WORD *)(a1 - 16) >> 6) & 0xF;
    v9 = (unsigned __int8 **)(a1 - result - 16);
  }
  v11 = &v9[v10];
  while ( v11 != v9 )
  {
    v12 = *v9;
    result = (unsigned int)**v9 - 5;
    if ( (unsigned __int8)(**v9 - 5) > 0x1Fu )
      goto LABEL_11;
    v13 = *(v12 - 16);
    if ( (v13 & 2) != 0 )
    {
      if ( *((_DWORD *)v12 - 6) <= 1u )
        goto LABEL_13;
      v14 = *((_QWORD *)v12 - 4);
    }
    else
    {
      v10 = (*((_WORD *)v12 - 8) >> 6) & 0xF;
      if ( ((*((_WORD *)v12 - 8) >> 6) & 0xFu) <= 1 )
        goto LABEL_13;
      v10 = (__int64)&v12[-8 * ((v13 >> 2) & 0xF)];
      v14 = v10 - 16;
    }
    result = *(_QWORD *)(v14 + 8);
    if ( result )
    {
      a4 = *(unsigned __int8 *)result;
      v10 = (unsigned int)(a4 - 5);
      if ( (unsigned __int8)(a4 - 5) <= 0x1Fu )
      {
        if ( a2 != result )
          goto LABEL_11;
        goto LABEL_14;
      }
    }
LABEL_13:
    result = 0;
    if ( a2 )
      goto LABEL_11;
LABEL_14:
    if ( !*(_BYTE *)(a3 + 28) )
      goto LABEL_23;
    result = *(_QWORD *)(a3 + 8);
    a4 = *(unsigned int *)(a3 + 20);
    v10 = result + 8 * a4;
    if ( result != v10 )
    {
      while ( v12 != *(unsigned __int8 **)result )
      {
        result += 8;
        if ( v10 == result )
          goto LABEL_18;
      }
      goto LABEL_11;
    }
LABEL_18:
    if ( (unsigned int)a4 < *(_DWORD *)(a3 + 16) )
    {
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a3 + 20) = a4;
      *(_QWORD *)v10 = v12;
      ++*(_QWORD *)a3;
    }
    else
    {
LABEL_23:
      result = (__int64)sub_C8CC70(a3, (__int64)v12, v10, a4, a5, a6);
    }
LABEL_11:
    ++v9;
  }
  return result;
}
