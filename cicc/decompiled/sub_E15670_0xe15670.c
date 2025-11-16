// Function: sub_E15670
// Address: 0xe15670
//
__int64 __fastcall sub_E15670(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r13
  _BYTE *v3; // r13
  _BYTE *v4; // r13
  __int64 result; // rax

  v2 = *(_BYTE **)(a1 + 16);
  if ( (char)(4 * v2[9]) >> 2 >= (unsigned int)((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 32LL))(v2, a2);
    if ( (v2[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v2 + 32LL))(*(_QWORD *)(a1 + 16));
    if ( (v2[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v2 + 40LL))(v2, a2);
  }
  sub_E12F20((__int64 *)a2, 3u, " ? ");
  v3 = *(_BYTE **)(a1 + 24);
  if ( (unsigned int)((char)(4 * v3[9]) >> 2) > 0x12 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 32LL))(v3, a2);
    if ( (v3[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v3 + 32LL))(*(_QWORD *)(a1 + 24), a2);
    if ( (v3[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v3 + 40LL))(v3, a2);
  }
  sub_E12F20((__int64 *)a2, 3u, " : ");
  v4 = *(_BYTE **)(a1 + 32);
  if ( (unsigned int)((char)(4 * v4[9]) >> 2) > 0x11 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 32LL))(v4, a2);
    if ( (v4[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v4 + 32LL))(*(_QWORD *)(a1 + 32), a2);
    result = v4[9] & 0xC0;
    if ( (v4[9] & 0xC0) != 0x40 )
      return (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v4 + 40LL))(v4, a2);
  }
  return result;
}
