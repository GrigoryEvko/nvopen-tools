// Function: sub_18DB8A0
// Address: 0x18db8a0
//
_QWORD *__fastcall sub_18DB8A0(__int64 a1, char a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rsi
  __int64 v4; // rdi
  _QWORD *result; // rax
  _QWORD *v6; // r8
  unsigned int v7; // r9d
  _QWORD *v8; // rcx

  sub_18DB890(*(_QWORD *)a1, a2);
  v2 = **(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(v2 + 16) == 29 )
  {
    v3 = (_QWORD *)sub_157EE30(**(_QWORD **)(a1 + 16));
    if ( !v3 || v3 == (_QWORD *)(**(_QWORD **)(a1 + 16) + 40LL) && (v3 = (_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL)) == 0 )
      BUG();
    if ( *((_BYTE *)v3 - 8) == 34 )
      *(_BYTE *)(*(_QWORD *)a1 + 136LL) = 1;
    v4 = *(_QWORD *)a1;
  }
  else
  {
    v3 = *(_QWORD **)(v2 + 32);
    v4 = *(_QWORD *)a1;
    if ( !v3 )
      goto LABEL_4;
  }
  v3 -= 3;
LABEL_4:
  result = *(_QWORD **)(v4 + 88);
  if ( *(_QWORD **)(v4 + 96) != result )
    return sub_16CCBA0(v4 + 80, (__int64)v3);
  v6 = &result[*(unsigned int *)(v4 + 108)];
  v7 = *(_DWORD *)(v4 + 108);
  if ( result == v6 )
  {
LABEL_22:
    if ( v7 >= *(_DWORD *)(v4 + 104) )
      return sub_16CCBA0(v4 + 80, (__int64)v3);
    *(_DWORD *)(v4 + 108) = v7 + 1;
    *v6 = v3;
    ++*(_QWORD *)(v4 + 80);
  }
  else
  {
    v8 = 0;
    while ( v3 != (_QWORD *)*result )
    {
      if ( *result == -2 )
        v8 = result;
      if ( v6 == ++result )
      {
        if ( !v8 )
          goto LABEL_22;
        *v8 = v3;
        --*(_DWORD *)(v4 + 112);
        ++*(_QWORD *)(v4 + 80);
        return result;
      }
    }
  }
  return result;
}
