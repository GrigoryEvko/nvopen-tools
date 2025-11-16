// Function: sub_2240CE0
// Address: 0x2240ce0
//
__int64 __fastcall sub_2240CE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v6; // rax
  size_t v7; // r9
  _BYTE *v8; // rdx
  _BYTE *v9; // rdi
  __int64 v10; // rdx
  __int64 result; // rax

  v4 = a2 + a3;
  v6 = a1[1];
  v7 = v6 - v4;
  if ( v6 != v4 && a3 )
  {
    v8 = (_BYTE *)(*a1 + v4);
    v9 = (_BYTE *)(a2 + *a1);
    if ( v7 == 1 )
      *v9 = *v8;
    else
      memmove(v9, v8, v7);
    v6 = a1[1];
  }
  v10 = *a1;
  result = v6 - a3;
  a1[1] = result;
  *(_BYTE *)(v10 + result) = 0;
  return result;
}
