// Function: sub_815F90
// Address: 0x815f90
//
__int64 __fastcall sub_815F90(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r12
  int v4; // eax

  result = *(_QWORD *)(a1 + 168);
  v3 = *(_QWORD *)(result + 8);
  if ( v3 )
  {
    result = *(unsigned __int8 *)(v3 + 140);
    if ( (unsigned __int8)(result - 9) <= 2u || (_BYTE)result == 2 && (*(_BYTE *)(v3 + 161) & 8) != 0 )
    {
      sub_815C30(v3);
      result = *(_QWORD *)(v3 + 8);
      if ( result )
      {
        *(_QWORD *)(a1 + 8) = result;
        *(_QWORD *)(a1 + 24) = *(_QWORD *)(v3 + 24);
        v4 = *(_BYTE *)(v3 + 89) & 8 | *(_BYTE *)(a1 + 89) & 0xF7;
        *(_BYTE *)(a1 + 89) = v4;
        result = *(_BYTE *)(v3 + 89) & 0x40 | v4 & 0xFFFFFFBF;
        *(_BYTE *)(a1 + 89) = result;
      }
      else
      {
        *(_DWORD *)(a2 + 48) = 1;
      }
    }
  }
  return result;
}
