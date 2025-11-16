// Function: sub_E22F00
// Address: 0xe22f00
//
__int64 __fastcall sub_E22F00(__int64 a1, size_t *a2)
{
  __int64 result; // rax
  size_t v3; // rdx
  _BYTE *v4; // rcx

  result = sub_E20730(a2, 2u, "_E");
  if ( !(_BYTE)result )
  {
    v3 = *a2;
    if ( *a2 && (v4 = (_BYTE *)a2[1], *v4 == 90) )
    {
      a2[1] = (size_t)(v4 + 1);
      *a2 = v3 - 1;
    }
    else
    {
      *(_BYTE *)(a1 + 8) = 1;
    }
  }
  return result;
}
