// Function: sub_1D047D0
// Address: 0x1d047d0
//
char *__fastcall sub_1D047D0(__int64 a1, __int64 a2)
{
  int v3; // eax
  char *result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_DWORD *)(a1 + 40);
  v6 = a2;
  result = (char *)(unsigned int)(v3 + 1);
  *(_DWORD *)(a1 + 40) = (_DWORD)result;
  *(_DWORD *)(a2 + 196) = (_DWORD)result;
  v5 = *(_BYTE **)(a1 + 24);
  if ( v5 == *(_BYTE **)(a1 + 32) )
    return sub_1CFD630(a1 + 16, v5, &v6);
  if ( v5 )
  {
    *(_QWORD *)v5 = a2;
    v5 = *(_BYTE **)(a1 + 24);
  }
  *(_QWORD *)(a1 + 24) = v5 + 8;
  return result;
}
