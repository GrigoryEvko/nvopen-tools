// Function: sub_8BFC10
// Address: 0x8bfc10
//
__int64 __fastcall sub_8BFC10(__int64 a1, __m128i *a2, __int64 a3)
{
  int v3; // r14d
  char v5; // al
  __int64 v6; // rbx
  __int64 result; // rax

  v3 = 0;
  v5 = *(_BYTE *)(a1 + 80);
  v6 = a1;
  if ( v5 == 17 )
  {
    v6 = *(_QWORD *)(a1 + 88);
    if ( !v6 )
      return 0;
    v5 = *(_BYTE *)(v6 + 80);
    v3 = 1;
  }
  if ( v5 == 20 )
    goto LABEL_6;
  while ( v3 )
  {
    v6 = *(_QWORD *)(v6 + 8);
    if ( !v6 )
      break;
    if ( *(_BYTE *)(v6 + 80) == 20 )
    {
LABEL_6:
      result = sub_8B8060(v6, a2, a3, 1, 0);
      if ( (_DWORD)result )
        return result;
    }
  }
  return 0;
}
