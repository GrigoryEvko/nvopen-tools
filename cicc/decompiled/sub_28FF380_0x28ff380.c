// Function: sub_28FF380
// Address: 0x28ff380
//
__int64 __fastcall sub_28FF380(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // eax
  __int64 (*v6)(); // rax
  unsigned int v7; // r8d
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 (*v11)(); // rax
  unsigned int v12; // r8d

  v5 = *(unsigned __int8 *)(a1 + 8);
  if ( (_BYTE)v5 == 14 )
  {
    v6 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
    if ( v6 == sub_BD8D60 )
      return 1;
    v9 = ((__int64 (__fastcall *)(__int64, __int64))v6)(a2, a1);
    a3 = v9;
    LOWORD(a3) = BYTE1(v9);
    if ( !BYTE1(v9) || (_BYTE)v9 )
      return 1;
    v5 = *(unsigned __int8 *)(a1 + 8);
  }
  v7 = 0;
  if ( (unsigned int)(v5 - 17) > 1 )
    return v7;
  v10 = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(v10 + 8) != 14 )
    return v7;
  v11 = *(__int64 (**)())(*(_QWORD *)a2 + 16LL);
  if ( v11 == sub_BD8D60 )
    return 1;
  v12 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD))v11)(a2, v10, a3, a4, 0);
  if ( !BYTE1(v12) )
    return 1;
  return v12;
}
