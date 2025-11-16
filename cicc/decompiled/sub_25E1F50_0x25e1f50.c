// Function: sub_25E1F50
// Address: 0x25e1f50
//
__int64 __fastcall sub_25E1F50(__int64 *a1, int a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx

  if ( !*(_BYTE *)(*a1 + 4) || (result = 0, a2 == *(_DWORD *)*a1) )
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD, __int64))a1[2])(*(_QWORD *)(a1[2] + 8), a3);
    result = sub_25E1B40(a3, a1[1], v5);
    if ( !(_BYTE)result )
    {
      v6 = *a1;
      *(_DWORD *)v6 = a2;
      *(_BYTE *)(v6 + 4) = 1;
    }
  }
  return result;
}
