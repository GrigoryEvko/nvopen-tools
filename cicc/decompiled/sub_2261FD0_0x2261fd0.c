// Function: sub_2261FD0
// Address: 0x2261fd0
//
__int64 __fastcall sub_2261FD0(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v4; // rax
  bool v5; // zf
  __int64 result; // rax

  v4 = *a3;
  *(_BYTE *)a1 = a2;
  v5 = *(_BYTE *)(v4 + 44) == 0;
  *(_DWORD *)(a1 + 4) = HIDWORD(a2);
  *(_DWORD *)(a1 + 8) = a4;
  if ( v5 )
    goto LABEL_2;
  result = sub_22F59B0(a3[1], *(unsigned __int16 *)(v4 + 56));
  if ( result )
  {
    v4 = sub_22F59B0(a3[1], *(unsigned __int16 *)(*a3 + 56));
LABEL_2:
    result = *(unsigned int *)(v4 + 40);
    *(_DWORD *)(a1 + 12) = result;
    return result;
  }
  *(_DWORD *)(a1 + 12) = 0;
  return result;
}
