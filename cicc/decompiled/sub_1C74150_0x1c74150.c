// Function: sub_1C74150
// Address: 0x1c74150
//
__int64 __fastcall sub_1C74150(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rax
  __int64 v5; // r8

  v2 = *(unsigned int *)(a1 + 56);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
  {
    v3 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v4 = &v3[3 * v2 + 1];
    if ( a2 != *v4 )
      goto LABEL_3;
    return v3[3];
  }
  v3 = *(_QWORD **)(a1 - 8);
  v4 = &v3[3 * v2 + 1];
  if ( a2 == *v4 )
    return v3[3];
LABEL_3:
  v5 = 0;
  if ( a2 == v4[1] )
    return *v3;
  return v5;
}
