// Function: sub_37B43A0
// Address: 0x37b43a0
//
__int64 __fastcall sub_37B43A0(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // rax
  int v4; // edx
  unsigned int *v5; // rdx
  int v7; // ecx
  __int64 v8; // rsi
  __int64 v9; // rdi
  _QWORD *v10; // rax
  _QWORD *v11; // rcx
  __int64 v12; // rcx

  if ( !a2 )
    return 0;
  v2 = *a2;
  if ( *a2 )
  {
    v4 = *(_DWORD *)(v2 + 64);
    if ( v4 )
    {
      v5 = (unsigned int *)(*(_QWORD *)(v2 + 40) + 40LL * (unsigned int)(v4 - 1));
      if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2]) == 262 )
        return 1;
    }
    if ( (v7 = *(_DWORD *)(v2 + 24), v7 >= 0)
      || (v12 = (unsigned int)~v7, (unsigned int)v12 <= 0x13) && ((1LL << v12) & 0x81700) != 0
      || (unsigned __int8)sub_37F0C40(a1[20], *(_QWORD *)(a1[18] + 8LL) - 40 * v12) )
    {
      v8 = a1[21];
      v9 = a1[22];
      if ( v9 == v8 )
        return 1;
      while ( 1 )
      {
        v10 = *(_QWORD **)(*(_QWORD *)v8 + 120LL);
        v11 = &v10[2 * *(unsigned int *)(*(_QWORD *)v8 + 128LL)];
        if ( v11 != v10 )
          break;
LABEL_12:
        v8 += 8;
        if ( v9 == v8 )
          return 1;
      }
      while ( (*v10 & 6) != 0 || a2 != (__int64 *)(*v10 & 0xFFFFFFFFFFFFFFF8LL) )
      {
        v10 += 2;
        if ( v11 == v10 )
          goto LABEL_12;
      }
    }
  }
  return 0;
}
