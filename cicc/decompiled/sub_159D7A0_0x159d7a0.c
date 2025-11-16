// Function: sub_159D7A0
// Address: 0x159d7a0
//
__int64 ****__fastcall sub_159D7A0(__int64 ***a1)
{
  __int64 *v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 ****result; // rax
  int v6; // r12d
  __int64 v7; // rax
  unsigned int v8; // edx
  __int64 ***v9; // rcx
  int v10; // eax
  int v11; // esi

  v1 = **a1;
  v2 = *v1;
  v3 = *(unsigned int *)(*v1 + 1800);
  v4 = *(_QWORD *)(*v1 + 1784);
  if ( (_DWORD)v3 )
  {
    v6 = v3 - 1;
    v7 = v6 & (unsigned int)sub_159D500((__int64)a1);
    v8 = v7;
    result = (__int64 ****)(v4 + 8 * v7);
    v9 = *result;
    if ( a1 == *result )
      goto LABEL_3;
    v10 = 1;
    while ( v9 != (__int64 ***)-8LL )
    {
      v11 = v10 + 1;
      v8 = v6 & (v10 + v8);
      result = (__int64 ****)(v4 + 8LL * v8);
      v9 = *result;
      if ( a1 == *result )
        goto LABEL_3;
      v10 = v11;
    }
    v4 = *(_QWORD *)(v2 + 1784);
    v3 = *(unsigned int *)(v2 + 1800);
  }
  result = (__int64 ****)(v4 + 8 * v3);
LABEL_3:
  *result = (__int64 ***)-16LL;
  --*(_DWORD *)(v2 + 1792);
  ++*(_DWORD *)(v2 + 1796);
  return result;
}
