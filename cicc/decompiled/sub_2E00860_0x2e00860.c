// Function: sub_2E00860
// Address: 0x2e00860
//
_QWORD *__fastcall sub_2E00860(__int64 a1, __int64 *a2)
{
  _QWORD *v2; // rbx
  unsigned int v3; // ecx
  unsigned int v4; // eax
  _QWORD *v5; // rdx
  bool v6; // cc
  _QWORD *result; // rax

  v2 = *(_QWORD **)(a1 + 16);
  if ( v2 )
  {
    v3 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
    while ( 1 )
    {
      v4 = *(_DWORD *)((v2[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v2[4] >> 1) & 3;
      v5 = (_QWORD *)v2[3];
      if ( v3 < v4 )
        v5 = (_QWORD *)v2[2];
      if ( !v5 )
        break;
      v2 = v5;
    }
    if ( v3 >= v4 )
      goto LABEL_8;
  }
  else
  {
    v2 = (_QWORD *)(a1 + 8);
  }
  result = 0;
  if ( v2 == *(_QWORD **)(a1 + 24) )
    return result;
  v2 = (_QWORD *)sub_220EF80((__int64)v2);
  v3 = *(_DWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a2 >> 1) & 3;
  v4 = *(_DWORD *)((v2[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v2[4] >> 1) & 3;
LABEL_8:
  v6 = v3 <= v4;
  result = v2;
  if ( !v6 )
    return 0;
  return result;
}
