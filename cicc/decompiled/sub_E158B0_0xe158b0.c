// Function: sub_E158B0
// Address: 0xe158b0
//
__int64 __fastcall sub_E158B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // rdx
  char v4; // r13
  _BYTE *v5; // r15
  char v6; // al
  bool v7; // zf
  bool v8; // bl
  _BYTE *v9; // r15
  __int64 result; // rax

  if ( *(_DWORD *)(a2 + 32) )
    goto LABEL_4;
  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_BYTE **)(a1 + 32);
  if ( v2 == 1 )
  {
    if ( *v3 != 62 )
    {
LABEL_4:
      v4 = 0;
      goto LABEL_5;
    }
  }
  else if ( v2 != 2 || *(_WORD *)v3 != 15934 )
  {
    goto LABEL_4;
  }
  v4 = 1;
  *(_DWORD *)(a2 + 32) = 1;
  sub_E14360(a2, 40);
LABEL_5:
  v5 = *(_BYTE **)(a1 + 16);
  v6 = (char)(4 * *(_BYTE *)(a1 + 9)) >> 2;
  v7 = v6 == 17;
  if ( v6 == 17 )
    v6 = 15;
  v8 = v7;
  if ( (char)(4 * v5[9]) >> 2 >= (unsigned int)!v7 + v6 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 32LL))(v5, a2);
    if ( (v5[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 40LL))(v5, a2);
    --*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 41);
    if ( *(_QWORD *)(a1 + 24) != 1 )
      goto LABEL_11;
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v5 + 32LL))(*(_QWORD *)(a1 + 16), a2);
    if ( (v5[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v5 + 40LL))(v5, a2);
    if ( *(_QWORD *)(a1 + 24) != 1 )
      goto LABEL_11;
  }
  if ( **(_BYTE **)(a1 + 32) != 44 )
LABEL_11:
    sub_E12F20((__int64 *)a2, 1u, " ");
  sub_E12F20((__int64 *)a2, *(_QWORD *)(a1 + 24), *(const void **)(a1 + 32));
  sub_E12F20((__int64 *)a2, 1u, " ");
  v9 = *(_BYTE **)(a1 + 40);
  if ( (char)(4 * v9[9]) >> 2 >= ((char)(4 * *(_BYTE *)(a1 + 9)) >> 2) + (unsigned int)v8 )
  {
    ++*(_DWORD *)(a2 + 32);
    sub_E14360(a2, 40);
    (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v9 + 32LL))(v9, a2);
    if ( (v9[9] & 0xC0) != 0x40 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v9 + 40LL))(v9, a2);
    --*(_DWORD *)(a2 + 32);
    result = sub_E14360(a2, 41);
  }
  else
  {
    (*(void (__fastcall **)(_QWORD, __int64))(*(_QWORD *)v9 + 32LL))(*(_QWORD *)(a1 + 40), a2);
    result = v9[9] & 0xC0;
    if ( (v9[9] & 0xC0) != 0x40 )
      result = (*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v9 + 40LL))(v9, a2);
  }
  if ( v4 )
  {
    --*(_DWORD *)(a2 + 32);
    return sub_E14360(a2, 41);
  }
  return result;
}
