// Function: sub_1BBCE00
// Address: 0x1bbce00
//
__int64 __fastcall sub_1BBCE00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r14d
  __int64 v5; // r15
  _QWORD *v7; // rax
  unsigned __int64 v8; // rcx
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-38h]

  if ( *(_BYTE *)(a2 + 8) == 13 )
  {
    v4 = *(_DWORD *)(a2 + 12);
    v11 = a2;
    v5 = **(_QWORD **)(a2 + 16);
  }
  else
  {
    v4 = *(_DWORD *)(a2 + 32);
    v5 = *(_QWORD *)(a2 + 24);
    v11 = 0;
  }
  if ( !(unsigned __int8)sub_1643F10(v5) )
    return 0;
  if ( (*(_BYTE *)(v5 + 8) & 0xFD) == 4 )
    return 0;
  v7 = sub_16463B0((__int64 *)v5, v4);
  v8 = (sub_127FA20(a3, (__int64)v7) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(unsigned int *)(a1 + 1396) > v8
    || *(unsigned int *)(a1 + 1392) < v8
    || v8 != ((sub_127FA20(a3, a2) + 7) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    return 0;
  }
  if ( v11 )
  {
    v9 = *(_QWORD **)(v11 + 16);
    v10 = &v9[*(unsigned int *)(v11 + 12)];
    if ( v10 != v9 )
    {
      while ( v5 == *v9 )
      {
        if ( v10 == ++v9 )
          return v4;
      }
      return 0;
    }
  }
  return v4;
}
