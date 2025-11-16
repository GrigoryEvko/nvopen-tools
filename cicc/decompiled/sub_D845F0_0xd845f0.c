// Function: sub_D845F0
// Address: 0xd845f0
//
__int64 __fastcall sub_D845F0(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d

  v1 = *(_QWORD *)(a1 + 8);
  v2 = 0;
  if ( v1 && *(_DWORD *)v1 == 2 && (v2 = (unsigned __int8)qword_4F87EC8, !(_BYTE)qword_4F87EC8) )
    return *(unsigned __int8 *)(v1 + 72);
  else
    return v2;
}
