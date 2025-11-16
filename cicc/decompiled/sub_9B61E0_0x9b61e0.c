// Function: sub_9B61E0
// Address: 0x9b61e0
//
__int64 __fastcall sub_9B61E0(__int64 a1, __int64 a2, char *a3, __int64 a4)
{
  unsigned int v5; // r14d
  int v6; // r8d
  char v7; // al

  if ( !a3[1] )
  {
    v5 = *(_DWORD *)(a4 + 24);
    if ( v5 <= 0x40 )
    {
      v7 = 1;
      if ( *(_QWORD *)(a4 + 16) )
        goto LABEL_4;
    }
    else
    {
      v6 = sub_C444A0(a4 + 16);
      v7 = 1;
      if ( v5 != v6 )
      {
LABEL_4:
        *a3 = v7;
        a3[1] = 1;
        return (unsigned __int8)*a3;
      }
    }
    v7 = sub_9A6530(a2, *(_QWORD *)a1, *(const __m128i **)(a1 + 8), **(_DWORD **)(a1 + 16));
    goto LABEL_4;
  }
  return (unsigned __int8)*a3;
}
