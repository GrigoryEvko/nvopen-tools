// Function: sub_1595A50
// Address: 0x1595a50
//
__int64 __fastcall sub_1595A50(__int64 a1, unsigned int a2)
{
  unsigned int *v2; // rbx
  unsigned int v3; // eax

  v2 = (unsigned int *)sub_1595950(a1, a2);
  v3 = *(_DWORD *)(sub_1595890(a1) + 8) >> 8;
  if ( v3 == 32 )
    return *v2;
  if ( v3 > 0x20 )
    return *(_QWORD *)v2;
  if ( v3 == 8 )
    return *(unsigned __int8 *)v2;
  return *(unsigned __int16 *)v2;
}
