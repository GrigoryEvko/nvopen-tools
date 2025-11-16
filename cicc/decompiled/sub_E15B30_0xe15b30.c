// Function: sub_E15B30
// Address: 0xe15b30
//
__int64 __fastcall sub_E15B30(_BYTE *a1, __int64 a2, int a3, unsigned __int8 a4)
{
  __int64 result; // rax

  if ( (char)(4 * a1[9]) >> 2 >= a3 + (unsigned int)a4 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 32LL))(a1, a2);
    if ( (a1[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 40LL))(a1, a2);
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)a1 + 32LL))(a1);
    result = a1[9] & 0xC0;
    if ( (a1[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)a1 + 40LL))(a1, a2);
  }
  return result;
}
