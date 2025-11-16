// Function: sub_C14AD0
// Address: 0xc14ad0
//
__int64 __fastcall sub_C14AD0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // r13
  _QWORD *v4; // r12
  unsigned int v5; // eax
  int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // rax

  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v2 = *(__int64 **)(a2 - 8);
    v3 = *v2;
    v4 = v2 + 3;
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v5 = sub_C92610(v4, v3);
  v6 = sub_C92860(a1 + 304, v4, v3, v5);
  if ( v6 == -1 )
    return 0;
  v7 = *(_QWORD *)(a1 + 304);
  v8 = v7 + 8LL * v6;
  if ( v8 == v7 + 8LL * *(unsigned int *)(a1 + 312) )
    return 0;
  else
    return *(unsigned int *)(*(_QWORD *)v8 + 8LL);
}
