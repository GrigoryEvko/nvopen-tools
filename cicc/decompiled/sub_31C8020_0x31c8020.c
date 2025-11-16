// Function: sub_31C8020
// Address: 0x31c8020
//
__int64 __fastcall sub_31C8020(__int64 *a1)
{
  __int64 v1; // r8
  __int64 result; // rax
  unsigned int v3; // ecx
  unsigned int v4; // edx

  v1 = *a1;
  while ( 1 )
  {
    result = *(a1 - 1);
    v3 = *(_DWORD *)(*(_QWORD *)(v1 + 56) + 72LL);
    v4 = *(_DWORD *)(*(_QWORD *)(result + 56) + 72LL);
    if ( v3 == v4 )
      break;
    if ( v3 >= v4 )
      goto LABEL_6;
LABEL_3:
    *a1-- = result;
  }
  if ( *(_DWORD *)(v1 + 64) < *(_DWORD *)(result + 64) )
    goto LABEL_3;
LABEL_6:
  *a1 = v1;
  return result;
}
