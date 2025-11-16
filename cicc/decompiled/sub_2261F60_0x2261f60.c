// Function: sub_2261F60
// Address: 0x2261f60
//
__int64 __fastcall sub_2261F60(__int64 a1, __int64 a2, int a3, __int64 *a4, int a5)
{
  __int64 v5; // rax
  bool v6; // zf
  __int64 result; // rax

  v5 = *a4;
  v6 = *(_BYTE *)(*a4 + 44) == 0;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 8) = a3;
  *(_DWORD *)(a1 + 12) = a5;
  if ( v6 )
    goto LABEL_2;
  result = sub_22F59B0(a4[1], *(unsigned __int16 *)(v5 + 56));
  if ( result )
  {
    v5 = sub_22F59B0(a4[1], *(unsigned __int16 *)(*a4 + 56));
LABEL_2:
    result = *(unsigned int *)(v5 + 40);
    *(_DWORD *)(a1 + 16) = result;
    return result;
  }
  *(_DWORD *)(a1 + 16) = 0;
  return result;
}
