// Function: sub_162FCD0
// Address: 0x162fcd0
//
char *__fastcall sub_162FCD0(
        __int64 a1,
        __int64 a2,
        unsigned __int8 *a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v13; // rsi
  unsigned __int8 *v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  double v18; // xmm4_8
  double v19; // xmm5_8
  char *result; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  char *v24; // r14
  int v25; // r13d
  unsigned int i; // ebx
  __int64 v27; // rdi

  v13 = (a2 - (a1 - 8LL * *(unsigned int *)(a1 + 8))) >> 3;
  if ( *(_BYTE *)(a1 + 1) )
    return (char *)sub_1623D00(a1, v13, (__int64)a3);
  sub_1621740(a1);
  v13 = (unsigned int)v13;
  v14 = *(unsigned __int8 **)(a1 + 8 * ((unsigned int)v13 - (unsigned __int64)*(unsigned int *)(a1 + 8)));
  sub_1623D00(a1, v13, (__int64)a3);
  if ( (unsigned __int8 *)a1 == a3 || v14 && !a3 && *v14 == 1 )
  {
    if ( *(_BYTE *)(a1 + 1) == 2 || (v16 = *(unsigned int *)(a1 + 12), (_DWORD)v16) )
      sub_161F180(a1, (unsigned int)v13, v15, v16, v17);
    return sub_1621390((char *)a1);
  }
  result = (char *)sub_162D4F0(a1, a4, a5, a6, a7, v18, v19, a10, a11);
  v24 = result;
  if ( (char *)a1 == result )
  {
    if ( *(_BYTE *)(a1 + 1) == 2 || *(_DWORD *)(a1 + 12) )
      return (char *)sub_161F190((__int64)result, v14, a3);
  }
  else
  {
    if ( *(_BYTE *)(a1 + 1) != 2 && !*(_DWORD *)(a1 + 12) )
      return sub_1621390((char *)a1);
    v25 = *(_DWORD *)(a1 + 8);
    if ( v25 )
    {
      for ( i = 0; i != v25; ++i )
      {
        v13 = i;
        sub_1623D00(a1, v13, 0);
      }
    }
    v27 = *(_QWORD *)(a1 + 16);
    if ( (v27 & 4) != 0 )
    {
      v13 = (__int64)v24;
      sub_16302D0(v27 & 0xFFFFFFFFFFFFFFF8LL, v24);
    }
    return (char *)sub_1623F10(a1, v13, v21, v22, v23);
  }
  return result;
}
