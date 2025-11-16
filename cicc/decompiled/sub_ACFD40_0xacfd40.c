// Function: sub_ACFD40
// Address: 0xacfd40
//
__int64 __fastcall sub_ACFD40(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r14
  int v5; // r13d
  __int64 *v6; // rdx
  __int64 v7; // rcx
  int v8; // edx
  int v9; // esi

  result = **(_QWORD **)(a1 + 8);
  v2 = *(_QWORD *)result;
  v3 = *(unsigned int *)(*(_QWORD *)result + 2144LL);
  v4 = *(_QWORD *)(*(_QWORD *)result + 2128LL);
  if ( !(_DWORD)v3 )
  {
LABEL_7:
    v6 = (__int64 *)(v4 + 8 * v3);
    goto LABEL_3;
  }
  v5 = v3 - 1;
  result = v5 & (unsigned int)sub_ACF990(a1);
  v6 = (__int64 *)(v4 + 8LL * (unsigned int)result);
  v7 = *v6;
  if ( a1 != *v6 )
  {
    v8 = 1;
    while ( v7 != -4096 )
    {
      v9 = v8 + 1;
      result = v5 & (unsigned int)(v8 + result);
      v6 = (__int64 *)(v4 + 8LL * (unsigned int)result);
      v7 = *v6;
      if ( a1 == *v6 )
        goto LABEL_3;
      v8 = v9;
    }
    v4 = *(_QWORD *)(v2 + 2128);
    v3 = *(unsigned int *)(v2 + 2144);
    goto LABEL_7;
  }
LABEL_3:
  *v6 = -8192;
  --*(_DWORD *)(v2 + 2136);
  ++*(_DWORD *)(v2 + 2140);
  return result;
}
