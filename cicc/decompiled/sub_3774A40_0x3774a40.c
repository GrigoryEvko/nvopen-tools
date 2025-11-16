// Function: sub_3774A40
// Address: 0x3774a40
//
__int64 __fastcall sub_3774A40(__int64 a1)
{
  int v1; // edx
  __int64 v2; // rax

  v1 = *(_DWORD *)(a1 + 24);
  if ( v1 > 239 )
  {
    v2 = (unsigned int)(v1 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v2 = 40;
    if ( v1 <= 237 )
      v2 = (unsigned int)(v1 - 101) < 0x30 ? 0x28 : 0;
  }
  return *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + v2) + 48LL)
                             + 16LL * *(unsigned int *)(*(_QWORD *)(a1 + 40) + v2 + 8));
}
