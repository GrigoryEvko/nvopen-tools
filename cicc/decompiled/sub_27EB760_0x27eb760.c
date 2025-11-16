// Function: sub_27EB760
// Address: 0x27eb760
//
__int64 __fastcall sub_27EB760(_QWORD *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  _BYTE *v5; // rax
  __int64 result; // rax
  _QWORD *v8; // rax

  if ( *(_DWORD *)(a3 + 8) > *(_DWORD *)(a3 + 4) )
  {
    v8 = sub_103DDE0(a1);
    result = (*(__int64 (__fastcall **)(_QWORD *, _BYTE *, __int64))(*v8 + 16LL))(v8, a4, a2);
    ++*(_DWORD *)(a3 + 4);
  }
  else
  {
    v5 = a4 - 64;
    if ( *a4 == 26 )
      v5 = a4 - 32;
    return *(_QWORD *)v5;
  }
  return result;
}
