// Function: sub_1B43580
// Address: 0x1b43580
//
unsigned __int64 __fastcall sub_1B43580(__int64 a1)
{
  __int64 *v1; // r15
  unsigned __int64 result; // rax
  __int64 *v3; // r12
  int v4; // r8d
  int v5; // r9d
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rbx
  __int64 v8; // rax

  v1 = *(__int64 **)a1;
  result = *(_QWORD *)(a1 + 8);
  *(_BYTE *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v3 = &v1[result];
  if ( v3 != v1 )
  {
    while ( 1 )
    {
      result = sub_157EBA0(*v1);
      if ( *(_QWORD *)(*(_QWORD *)(result + 40) + 48LL) == result + 24 )
        break;
      v6 = *(_QWORD *)(result + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v6 )
        break;
      while ( 1 )
      {
        v7 = v6 - 24;
        if ( *(_BYTE *)(v7 + 16) != 78 )
          break;
        v8 = *(_QWORD *)(v7 - 24);
        if ( *(_BYTE *)(v8 + 16) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v8 + 36) - 35) > 3 )
          break;
        result = v7 + 24;
        if ( *(_QWORD *)(*(_QWORD *)(v7 + 40) + 48LL) != v7 + 24 )
        {
          v6 = *(_QWORD *)(v7 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v6 )
            continue;
        }
        goto LABEL_14;
      }
      result = *(unsigned int *)(a1 + 24);
      if ( (unsigned int)result >= *(_DWORD *)(a1 + 28) )
      {
        sub_16CD150(a1 + 16, (const void *)(a1 + 32), 0, 8, v4, v5);
        result = *(unsigned int *)(a1 + 24);
      }
      ++v1;
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * result) = v7;
      ++*(_DWORD *)(a1 + 24);
      if ( v3 == v1 )
        return result;
    }
LABEL_14:
    *(_BYTE *)(a1 + 64) = 1;
  }
  return result;
}
