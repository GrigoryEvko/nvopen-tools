// Function: sub_30D5340
// Address: 0x30d5340
//
const char *__fastcall sub_30D5340(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdi
  __int64 v4; // rax
  int v5; // ecx
  const char *result; // rax

  sub_30D2590(a1, *(_QWORD *)(a1 + 96), *(_QWORD *)(a1 + 72));
  *(_DWORD *)(a1 + 704) += *(_DWORD *)(a1 + 660) + *(_DWORD *)(a1 + 656);
  v2 = sub_30D4FE0(*(__int64 **)(a1 + 8), *(unsigned __int8 **)(a1 + 96), *(_QWORD *)(a1 + 80));
  v3 = *(_QWORD *)(a1 + 72);
  v4 = *(int *)(a1 + 716) + (__int64)-v2;
  if ( v4 > 0x7FFFFFFF )
    v4 = 0x7FFFFFFF;
  if ( v4 < (__int64)0xFFFFFFFF80000000LL )
    LODWORD(v4) = 0x80000000;
  *(_DWORD *)(a1 + 716) = v4;
  v5 = v4;
  if ( ((*(_WORD *)(v3 + 2) >> 4) & 0x3FF) == 9 )
  {
    v5 = v4 + 2000;
    *(_DWORD *)(a1 + 716) = v4 + 2000;
  }
  if ( *(_DWORD *)(a1 + 704) > v5 || (result = "high cost", *(_BYTE *)(a1 + 648)) )
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 664) + 66LL) || !sub_B2DCC0(v3) )
      return 0;
    else
      return "delaying inlining for consideration in later phase";
  }
  return result;
}
