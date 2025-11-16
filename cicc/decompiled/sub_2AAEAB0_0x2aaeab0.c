// Function: sub_2AAEAB0
// Address: 0x2aaeab0
//
__int64 __fastcall sub_2AAEAB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // eax

  if ( (unsigned __int8)sub_B2D610(a1, 47) || (unsigned __int8)sub_B2D610(a1, 18) )
    return 1;
  if ( sub_11F3070(**(_QWORD **)(a2 + 32), a4, a5) )
  {
    v14 = *(_DWORD *)(a3 + 40);
    if ( v14 == -1 )
    {
      if ( (unsigned __int8)sub_F6E590(*(_QWORD *)(a3 + 104), a4, v10, v11, v12, v13) )
        return 1;
      v14 = *(_DWORD *)(a3 + 40);
    }
    if ( v14 == 1 )
      goto LABEL_8;
    return 1;
  }
LABEL_8:
  if ( (unsigned int)sub_23DF0D0(dword_500E728) && (unsigned int)dword_500E7A8 <= 2 )
    return dword_439F0A8[dword_500E7A8];
  result = *(unsigned int *)(a3 + 72);
  if ( (_DWORD)result )
  {
    if ( (_DWORD)result == 1 )
      return 3;
    else
      return (unsigned __int8)sub_DF9CC0(a6) != 0 ? 3 : 0;
  }
  return result;
}
