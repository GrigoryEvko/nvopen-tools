// Function: sub_2E026D0
// Address: 0x2e026d0
//
_QWORD *__fastcall sub_2E026D0(_QWORD *a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // eax
  unsigned int v5; // ebx
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v9; // rax

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5]
      || (*(_DWORD *)((*(_QWORD *)(a1[4] + 32LL) & 0xFFFFFFFFFFFFFFF8LL) + 24)
        | (unsigned int)(*(__int64 *)(a1[4] + 32LL) >> 1) & 3) >= (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                 | (unsigned int)(*a3 >> 1) & 3) )
    {
      return sub_2E00860((__int64)a1, a3);
    }
  }
  else
  {
    v4 = *(_DWORD *)((*(_QWORD *)(a2 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(a2 + 32) >> 1) & 3;
    v5 = *(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a3 >> 1) & 3;
    if ( v5 < v4 )
    {
      v6 = a1[3];
      if ( v6 == a2 )
        return (_QWORD *)v6;
      v7 = sub_220EF80(a2);
      if ( v5 > (*(_DWORD *)((*(_QWORD *)(v7 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
               | (unsigned int)(*(__int64 *)(v7 + 32) >> 1) & 3) )
      {
        v6 = 0;
        if ( *(_QWORD *)(v7 + 24) )
          return (_QWORD *)a2;
        return (_QWORD *)v6;
      }
      return sub_2E00860((__int64)a1, a3);
    }
    if ( v5 <= v4 )
      return (_QWORD *)a2;
    if ( a1[4] != a2 )
    {
      v9 = sub_220EEE0(a2);
      if ( v5 < (*(_DWORD *)((*(_QWORD *)(v9 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24)
               | (unsigned int)(*(__int64 *)(v9 + 32) >> 1) & 3) )
      {
        v6 = 0;
        if ( *(_QWORD *)(a2 + 24) )
          return (_QWORD *)v9;
        return (_QWORD *)v6;
      }
      return sub_2E00860((__int64)a1, a3);
    }
  }
  return 0;
}
