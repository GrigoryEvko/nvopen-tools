// Function: sub_984BF0
// Address: 0x984bf0
//
__int64 __fastcall sub_984BF0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 v5; // rcx
  _QWORD *v6; // rdx
  char v7; // dl
  __int64 v8; // r12

  v3 = *a1;
  if ( !*(_BYTE *)(v3 + 28) )
    goto LABEL_8;
  result = *(_QWORD *)(v3 + 8);
  v5 = *(unsigned int *)(v3 + 20);
  v6 = (_QWORD *)(result + 8 * v5);
  if ( (_QWORD *)result == v6 )
  {
LABEL_7:
    if ( (unsigned int)v5 < *(_DWORD *)(v3 + 16) )
    {
      *(_DWORD *)(v3 + 20) = v5 + 1;
      *v6 = a2;
      ++*(_QWORD *)v3;
LABEL_9:
      v8 = a1[1];
      result = *(unsigned int *)(v8 + 8);
      if ( result + 1 > (unsigned __int64)*(unsigned int *)(v8 + 12) )
      {
        sub_C8D5F0(v8, v8 + 16, result + 1, 8);
        result = *(unsigned int *)(v8 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v8 + 8 * result) = a2;
      ++*(_DWORD *)(v8 + 8);
      return result;
    }
LABEL_8:
    result = sub_C8CC70(v3, a2);
    if ( !v7 )
      return result;
    goto LABEL_9;
  }
  while ( a2 != *(_QWORD *)result )
  {
    result += 8;
    if ( v6 == (_QWORD *)result )
      goto LABEL_7;
  }
  return result;
}
