// Function: sub_222F680
// Address: 0x222f680
//
__int64 __fastcall sub_222F680(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 result; // rax
  unsigned int v5; // edx
  __int64 (__fastcall *v6)(__int64, unsigned int); // rax

  result = *(unsigned __int8 *)(a1 + (unsigned __int8)a2 + 313);
  if ( !(_BYTE)result )
  {
    v5 = a2;
    v6 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)a1 + 64LL);
    if ( v6 != sub_2216C50 )
      v5 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v6)(a1, (unsigned int)(char)a2, (unsigned int)(char)a3);
    result = a3;
    if ( (_BYTE)a3 != (_BYTE)v5 )
    {
      *(_BYTE *)(a1 + (unsigned __int8)a2 + 313) = v5;
      return v5;
    }
  }
  return result;
}
