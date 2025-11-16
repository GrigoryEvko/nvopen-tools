// Function: sub_5E4DE0
// Address: 0x5e4de0
//
__int64 __fastcall sub_5E4DE0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 result; // rax

  v5 = *a1;
  if ( !*a1 )
  {
    v5 = a1[24];
    if ( !v5 )
      goto LABEL_10;
  }
  while ( *(_BYTE *)(v5 + 80) != 8 || (*(_BYTE *)(*(_QWORD *)(v5 + 88) + 145LL) & 0x20) == 0 )
  {
    v5 = *(_QWORD *)(v5 + 16);
    if ( !v5 )
      goto LABEL_10;
  }
  if ( a2 != v5 )
    sub_6854C0(2425, a3, v5);
  else
LABEL_10:
    sub_6851C0(2915, a3);
  result = *(_QWORD *)(a2 + 88);
  *(_BYTE *)(result + 145) &= ~0x20u;
  return result;
}
