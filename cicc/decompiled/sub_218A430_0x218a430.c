// Function: sub_218A430
// Address: 0x218a430
//
_QWORD *__fastcall sub_218A430(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  _QWORD *result; // rax
  _BYTE *v9; // rax
  _WORD *v10; // rdx
  unsigned int v11; // edx
  __int64 v12; // rcx

  sub_21897A0(a1, a2, a3, a4);
  if ( a5 && !strcmp(a5, "add") )
  {
    v10 = *(_WORD **)(a4 + 24);
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)v10 <= 1u )
    {
      sub_16E7EE0(a4, ", ", 2u);
    }
    else
    {
      *v10 = 8236;
      *(_QWORD *)(a4 + 24) += 2LL;
    }
    v11 = a3 + 1;
    v12 = a4;
    return sub_21897A0(a1, a2, v11, v12);
  }
  result = (_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * (a3 + 1));
  if ( *(_BYTE *)result != 2 || result[1] )
  {
    v9 = *(_BYTE **)(a4 + 24);
    if ( *(_BYTE **)(a4 + 16) == v9 )
    {
      sub_16E7EE0(a4, "+", 1u);
    }
    else
    {
      *v9 = 43;
      ++*(_QWORD *)(a4 + 24);
    }
    v12 = a4;
    v11 = a3 + 1;
    return sub_21897A0(a1, a2, v11, v12);
  }
  return result;
}
