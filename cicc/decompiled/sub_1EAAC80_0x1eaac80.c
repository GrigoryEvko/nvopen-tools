// Function: sub_1EAAC80
// Address: 0x1eaac80
//
char *__fastcall sub_1EAAC80(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 (*v3)(void); // rdx
  char *result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1[276];
  v3 = *(__int64 (**)(void))(*(_QWORD *)v2 + 96LL);
  if ( (char *)v3 == (char *)sub_1D123B0 )
  {
    result = *(char **)(*(_QWORD *)v2 + 80LL);
    if ( result != (char *)nullsub_683 )
      result = (char *)((__int64 (*)(void))result)();
  }
  else
  {
    result = (char *)v3();
  }
  v6 = 0;
  v5 = (_BYTE *)a1[280];
  if ( v5 == (_BYTE *)a1[281] )
    return sub_1D12610((__int64)(a1 + 279), v5, &v6);
  if ( v5 )
  {
    *(_QWORD *)v5 = 0;
    v5 = (_BYTE *)a1[280];
  }
  a1[280] = v5 + 8;
  return result;
}
