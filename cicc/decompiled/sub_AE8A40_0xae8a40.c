// Function: sub_AE8A40
// Address: 0xae8a40
//
_QWORD *__fastcall sub_AE8A40(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *result; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  char v7; // dl
  unsigned __int8 v8; // al
  __int64 v9; // r13
  unsigned __int8 **v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // r13

  if ( !*(_BYTE *)(a1 + 428) )
    goto LABEL_8;
  result = *(_QWORD **)(a1 + 408);
  v5 = *(unsigned int *)(a1 + 420);
  v6 = &result[v5];
  if ( result == v6 )
  {
LABEL_7:
    if ( (unsigned int)v5 < *(_DWORD *)(a1 + 416) )
    {
      *(_DWORD *)(a1 + 420) = v5 + 1;
      *v6 = a3;
      ++*(_QWORD *)(a1 + 400);
LABEL_9:
      v8 = *(_BYTE *)(a3 - 16);
      v9 = a3 - 16;
      if ( (v8 & 2) != 0 )
        v10 = *(unsigned __int8 ***)(a3 - 32);
      else
        v10 = (unsigned __int8 **)(v9 - 8LL * ((v8 >> 2) & 0xF));
      sub_AE8080(a1, *v10);
      v11 = *(_BYTE *)(a3 - 16);
      if ( (v11 & 2) != 0 )
        v12 = *(_QWORD *)(a3 - 32);
      else
        v12 = v9 - 8LL * ((v11 >> 2) & 0xF);
      return (_QWORD *)sub_AE8230(a1, *(unsigned __int8 **)(v12 + 24));
    }
LABEL_8:
    result = (_QWORD *)sub_C8CC70(a1 + 400, a3);
    if ( !v7 )
      return result;
    goto LABEL_9;
  }
  while ( a3 != *result )
  {
    if ( v6 == ++result )
      goto LABEL_7;
  }
  return result;
}
