// Function: sub_6E6990
// Address: 0x6e6990
//
__int64 __fastcall sub_6E6990(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v2; // eax
  __int64 v3; // rcx
  unsigned int v4; // eax
  __int64 result; // rax

  *(_OWORD *)a1 = 0;
  v1 = dword_4F077BC;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 32) = 0;
  if ( v1 )
  {
    v2 = *(unsigned __int8 *)(a1 + 42);
    if ( qword_4F077A8 <= 0x9F5Fu )
    {
      v2 |= 1u;
      *(_BYTE *)(a1 + 42) = v2;
    }
  }
  else
  {
    v2 = *(unsigned __int8 *)(a1 + 42);
  }
  v3 = qword_4D03C50;
  if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
  {
    *(_BYTE *)(a1 + 40) |= 4u;
    v2 = *(_BYTE *)(v3 + 19) & 0x40 | v2 & 0xFFFFFFBF;
    *(_BYTE *)(a1 + 42) = v2;
  }
  v4 = (16 * ((*(_BYTE *)(v3 + 17) & 1) == 0)) | v2 & 0xFFFFFFEF;
  *(_BYTE *)(a1 + 42) = v4;
  result = (32 * (((*(_BYTE *)(v3 + 17) >> 1) ^ 1) & 1)) | v4 & 0xFFFFFFDF;
  *(_BYTE *)(a1 + 42) = result;
  return result;
}
