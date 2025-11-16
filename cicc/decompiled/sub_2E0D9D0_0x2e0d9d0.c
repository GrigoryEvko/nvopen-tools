// Function: sub_2E0D9D0
// Address: 0x2e0d9d0
//
_QWORD *__fastcall sub_2E0D9D0(_QWORD *a1, _QWORD *a2, __int64 *a3)
{
  unsigned int v4; // r13d
  unsigned int v5; // eax
  _QWORD *v6; // rdx
  unsigned int v7; // eax
  _QWORD *result; // rax
  unsigned int v9; // edx
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // eax

  if ( a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_2E0D880((__int64)a1, a3);
    v13 = a1[4];
    v14 = *a3;
    v15 = *(_DWORD *)((*(_QWORD *)(v13 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v13 + 32) >> 1) & 3;
    v16 = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v14 >> 1) & 3;
    if ( v15 >= v16
      && (v15 > v16
       || (*(_DWORD *)((*(_QWORD *)(v13 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
         | (unsigned int)(*(__int64 *)(v13 + 40) >> 1) & 3) >= (*(_DWORD *)((a3[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                              | (unsigned int)(a3[1] >> 1) & 3)) )
    {
      return sub_2E0D880((__int64)a1, a3);
    }
    return 0;
  }
  v4 = *(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a3 >> 1) & 3;
  v5 = *(_DWORD *)((a2[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)a2[4] >> 1) & 3;
  if ( v4 < v5 )
    goto LABEL_3;
  if ( v4 <= v5 )
  {
    v9 = *(_DWORD *)((a3[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3[1] >> 1) & 3;
    v10 = *(_DWORD *)((a2[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)a2[5] >> 1) & 3;
    if ( v9 < v10 )
    {
LABEL_3:
      if ( (_QWORD *)a1[3] == a2 )
        return a2;
      v6 = (_QWORD *)sub_220EF80((__int64)a2);
      v7 = *(_DWORD *)((v6[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | ((__int64)v6[4] >> 1) & 3;
      if ( v4 > v7
        || v4 >= v7
        && (*(_DWORD *)((v6[5] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)v6[5] >> 1) & 3) < (*(_DWORD *)((a3[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3[1] >> 1) & 3) )
      {
        result = 0;
        if ( v6[3] )
          return a2;
        return result;
      }
      return sub_2E0D880((__int64)a1, a3);
    }
    if ( v9 <= v10 )
      return a2;
  }
  if ( (_QWORD *)a1[4] == a2 )
    return 0;
  v11 = sub_220EEE0((__int64)a2);
  v12 = *(_DWORD *)((*(_QWORD *)(v11 + 32) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*(__int64 *)(v11 + 32) >> 1) & 3;
  if ( v4 >= v12
    && (v4 > v12
     || (*(_DWORD *)((a3[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a3[1] >> 1) & 3) >= (*(_DWORD *)((*(_QWORD *)(v11 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*(__int64 *)(v11 + 40) >> 1)
                                                                                              & 3)) )
  {
    return sub_2E0D880((__int64)a1, a3);
  }
  result = 0;
  if ( a2[3] )
    return (_QWORD *)v11;
  return result;
}
