// Function: sub_227B070
// Address: 0x227b070
//
_QWORD *__fastcall sub_227B070(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // ecx
  __int64 *v5; // rdx
  __int64 v6; // r10
  _QWORD *result; // rax
  int v8; // edx
  int v9; // r11d

  v2 = *(unsigned int *)(a1 + 72);
  v3 = *(_QWORD *)(a1 + 56);
  if ( !(_DWORD)v2 )
    return 0;
  v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v5 = (__int64 *)(v3 + 16LL * v4);
  v6 = *v5;
  if ( a2 != *v5 )
  {
    v8 = 1;
    while ( v6 != -4096 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        goto LABEL_3;
      v8 = v9;
    }
    return 0;
  }
LABEL_3:
  if ( v5 == (__int64 *)(v3 + 16 * v2) )
    return 0;
  result = (_QWORD *)(*(_QWORD *)a1 + 8LL * *((int *)v5 + 2));
  if ( (*result & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    return 0;
  if ( !*(_QWORD *)(*result & 0xFFFFFFFFFFFFFFF8LL) )
    return 0;
  return result;
}
