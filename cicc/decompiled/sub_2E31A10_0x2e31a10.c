// Function: sub_2E31A10
// Address: 0x2e31a10
//
unsigned __int64 __fastcall sub_2E31A10(__int64 a1, char a2)
{
  _QWORD *v2; // r8
  _QWORD *v3; // rdi
  _QWORD *v4; // rcx
  unsigned __int64 result; // rax
  __int16 v6; // dx

  v2 = *(_QWORD **)(a1 + 56);
  v3 = (_QWORD *)(a1 + 48);
  v4 = v3;
  if ( v2 == v3 )
    return (unsigned __int64)v3;
  while ( 1 )
  {
    result = *v4 & 0xFFFFFFFFFFFFFFF8LL;
    v6 = *(_WORD *)(result + 68);
    v4 = (_QWORD *)result;
    if ( (unsigned __int16)(v6 - 14) > 4u && (*(_BYTE *)(result + 44) & 4) == 0 && (v6 != 24 || !a2) )
      break;
    if ( v2 == (_QWORD *)result )
      return (unsigned __int64)v3;
  }
  return result;
}
