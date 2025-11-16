// Function: sub_310F6F0
// Address: 0x310f6f0
//
_DWORD *__fastcall sub_310F6F0(__int64 a1, __int64 a2, int a3, int a4, __int64 a5, __int64 a6)
{
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  __int64 v12; // rdx
  unsigned __int64 v13; // r13
  char *v14; // rcx
  __int64 v15; // r13
  _DWORD *v16; // rcx
  _DWORD *result; // rax
  __int64 v18; // rdx

  *(_QWORD *)a1 = a1 + 16;
  v10 = *(_QWORD *)(a2 + 8);
  v11 = *(_BYTE **)a2;
  sub_11F4570((__int64 *)a1, v11, (__int64)&v11[v10]);
  *(_DWORD *)(a1 + 32) = a3;
  *(_DWORD *)(a1 + 36) = a4;
  v13 = *(_QWORD *)(a6 + 8) - *(_QWORD *)a6;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  if ( v13 )
  {
    if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, v11, v12);
    v14 = (char *)sub_22077B0(v13);
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  *(_QWORD *)(a1 + 40) = v14;
  *(_QWORD *)(a1 + 56) = &v14[v13];
  *(_QWORD *)(a1 + 48) = v14;
  v15 = *(_QWORD *)(a6 + 8) - *(_QWORD *)a6;
  if ( *(_QWORD *)(a6 + 8) != *(_QWORD *)a6 )
    v14 = (char *)memmove(v14, *(const void **)a6, *(_QWORD *)(a6 + 8) - *(_QWORD *)a6);
  *(_QWORD *)(a1 + 48) = &v14[v15];
  v16 = *(_DWORD **)(a6 + 8);
  result = *(_DWORD **)a6;
  if ( *(_DWORD **)a6 == v16 )
  {
    v18 = 1;
  }
  else
  {
    LODWORD(v18) = 1;
    do
    {
      LODWORD(v18) = *result * v18;
      result += 2;
    }
    while ( v16 != result );
    v18 = (int)v18;
  }
  *(_QWORD *)(a1 + 72) = a5;
  *(_QWORD *)(a1 + 64) = v18;
  return result;
}
