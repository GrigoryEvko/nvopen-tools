// Function: sub_F8F540
// Address: 0xf8f540
//
unsigned __int64 __fastcall sub_F8F540(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int64 result; // rax
  _QWORD *v5; // rdx
  __int64 v6; // rcx

  v3 = 0;
  if ( (a1[7] & 0x20) != 0 )
    v3 = sub_B91C10((__int64)a1, 2);
  result = sub_BC8A80(v3, a2);
  if ( *a1 == 31 )
  {
    result = *(_WORD *)(*((_QWORD *)a1 - 12) + 2LL) & 0x3F;
    if ( (*(_WORD *)(*((_QWORD *)a1 - 12) + 2LL) & 0x3F) == 0x20 )
    {
      result = *(_QWORD *)a2;
      v5 = (_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - 8);
      v6 = **(_QWORD **)a2;
      **(_QWORD **)a2 = *v5;
      *v5 = v6;
    }
  }
  return result;
}
