// Function: sub_AAF450
// Address: 0xaaf450
//
__int64 __fastcall sub_AAF450(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // eax
  __int64 result; // rax
  unsigned int v5; // eax

  v2 = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 8) = v2;
  if ( v2 > 0x40 )
  {
    sub_C43780(a1, a2);
    v5 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v5;
    if ( v5 <= 0x40 )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a2;
    v3 = *(_DWORD *)(a2 + 24);
    *(_DWORD *)(a1 + 24) = v3;
    if ( v3 <= 0x40 )
    {
LABEL_3:
      result = *(_QWORD *)(a2 + 16);
      *(_QWORD *)(a1 + 16) = result;
      return result;
    }
  }
  return sub_C43780(a1 + 16, a2 + 16);
}
