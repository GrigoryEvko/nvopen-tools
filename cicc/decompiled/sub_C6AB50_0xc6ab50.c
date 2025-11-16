// Function: sub_C6AB50
// Address: 0xc6ab50
//
_BYTE *__fastcall sub_C6AB50(__int64 a1)
{
  unsigned __int64 v2; // rcx
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  int v5; // eax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  _BYTE *result; // rax

  sub_C6AAB0(a1);
  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 12);
  if ( v2 >= v3 )
  {
    if ( v3 < v2 + 1 )
    {
      sub_C8D5F0(a1, a1 + 16, v2 + 1, 8);
      v2 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v2) = 0;
    v4 = *(_QWORD *)a1;
    v7 = (unsigned int)(*(_DWORD *)(a1 + 8) + 1);
    *(_DWORD *)(a1 + 8) = v7;
  }
  else
  {
    v4 = *(_QWORD *)a1;
    v5 = *(_DWORD *)(a1 + 8);
    v6 = *(_QWORD *)a1 + 8 * v2;
    if ( v6 )
    {
      *(_DWORD *)v6 = 0;
      *(_BYTE *)(v6 + 4) = 0;
      v5 = *(_DWORD *)(a1 + 8);
      v4 = *(_QWORD *)a1;
    }
    v7 = (unsigned int)(v5 + 1);
    *(_DWORD *)(a1 + 8) = v7;
  }
  *(_DWORD *)(v4 + 8 * v7 - 8) = 1;
  v8 = *(_QWORD *)(a1 + 160);
  *(_DWORD *)(a1 + 172) += *(_DWORD *)(a1 + 168);
  result = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v8 + 24) )
    return (_BYTE *)sub_CB5D20(v8, 91);
  *(_QWORD *)(v8 + 32) = result + 1;
  *result = 91;
  return result;
}
