// Function: sub_154DEB0
// Address: 0x154deb0
//
_BYTE *__fastcall sub_154DEB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v5; // r14
  __int64 *v6; // r15
  __int64 v7; // r14
  __int64 v8; // rsi
  _WORD *v9; // rdx
  _BYTE *result; // rax
  _BYTE *v11; // rax

  if ( (*(_DWORD *)(a2 + 8) & 0x100) == 0 )
    return (_BYTE *)sub_1263B40(a3, "opaque");
  if ( ((*(_DWORD *)(a2 + 8) >> 8) & 2) != 0 )
  {
    v11 = *(_BYTE **)(a3 + 24);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(a3 + 16) )
    {
      sub_16E7DE0(a3, 60);
    }
    else
    {
      *(_QWORD *)(a3 + 24) = v11 + 1;
      *v11 = 60;
    }
  }
  if ( !*(_DWORD *)(a2 + 12) )
  {
    result = (_BYTE *)sub_1263B40(a3, "{}");
    if ( (*(_BYTE *)(a2 + 9) & 2) == 0 )
      return result;
    goto LABEL_13;
  }
  v5 = *(__int64 **)(a2 + 16);
  sub_1263B40(a3, "{ ");
  v6 = v5 + 1;
  sub_154DAA0(a1, *v5, a3);
  v7 = *(_QWORD *)(a2 + 16) + 8LL * *(unsigned int *)(a2 + 12);
  while ( v6 != (__int64 *)v7 )
  {
    v9 = *(_WORD **)(a3 + 24);
    if ( *(_QWORD *)(a3 + 16) - (_QWORD)v9 > 1u )
    {
      *v9 = 8236;
      *(_QWORD *)(a3 + 24) += 2LL;
    }
    else
    {
      sub_16E7EE0(a3, ", ", 2);
    }
    v8 = *v6++;
    sub_154DAA0(a1, v8, a3);
  }
  result = (_BYTE *)sub_1263B40(a3, " }");
  if ( (*(_BYTE *)(a2 + 9) & 2) != 0 )
  {
LABEL_13:
    result = *(_BYTE **)(a3 + 24);
    if ( (unsigned __int64)result >= *(_QWORD *)(a3 + 16) )
    {
      return (_BYTE *)sub_16E7DE0(a3, 62);
    }
    else
    {
      *(_QWORD *)(a3 + 24) = result + 1;
      *result = 62;
    }
  }
  return result;
}
