// Function: sub_12D6100
// Address: 0x12d6100
//
__int64 __fastcall sub_12D6100(__int64 a1, __int64 a2, __int64 *a3, int a4)
{
  __int64 v4; // rax
  bool v5; // zf
  __int64 result; // rax

  v4 = *a3;
  *(_BYTE *)a1 = a2;
  v5 = *(_BYTE *)(v4 + 36) == 0;
  *(_DWORD *)(a1 + 4) = HIDWORD(a2);
  *(_DWORD *)(a1 + 8) = a4;
  if ( v5 )
    goto LABEL_2;
  result = sub_1691920(a3[1], *(unsigned __int16 *)(v4 + 40));
  if ( result )
  {
    v4 = sub_1691920(a3[1], *(unsigned __int16 *)(*a3 + 40));
LABEL_2:
    result = *(unsigned int *)(v4 + 32);
    *(_DWORD *)(a1 + 12) = result;
    return result;
  }
  *(_DWORD *)(a1 + 12) = 0;
  return result;
}
