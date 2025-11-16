// Function: sub_12D6090
// Address: 0x12d6090
//
__int64 __fastcall sub_12D6090(__int64 a1, __int64 a2, int a3, __int64 *a4, int a5)
{
  __int64 v5; // rax
  bool v6; // zf
  __int64 result; // rax

  v5 = *a4;
  v6 = *(_BYTE *)(*a4 + 36) == 0;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 8) = a3;
  *(_DWORD *)(a1 + 12) = a5;
  if ( v6 )
    goto LABEL_2;
  result = sub_1691920(a4[1], *(unsigned __int16 *)(v5 + 40));
  if ( result )
  {
    v5 = sub_1691920(a4[1], *(unsigned __int16 *)(*a4 + 40));
LABEL_2:
    result = *(unsigned int *)(v5 + 32);
    *(_DWORD *)(a1 + 16) = result;
    return result;
  }
  *(_DWORD *)(a1 + 16) = 0;
  return result;
}
