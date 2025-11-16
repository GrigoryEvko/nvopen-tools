// Function: sub_2167B00
// Address: 0x2167b00
//
__int64 __fastcall sub_2167B00(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 (__fastcall *v5)(__int64, __int64, __int64); // rax
  unsigned int v6; // r12d
  __int64 savedregs; // [rsp+20h] [rbp+0h] BYREF

  v4 = *(_QWORD *)(a1 + 24);
  if ( a2 != 36 )
  {
    if ( a2 != 37 )
    {
      savedregs = (__int64)&savedregs;
      switch ( a2 )
      {
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
          JUMPOUT(0x14A1D50);
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35:
        case 38:
        case 39:
        case 40:
        case 41:
        case 42:
        case 43:
        case 44:
          JUMPOUT(0x14A1D38);
        case 45:
          JUMPOUT(0x14A1D90);
        case 46:
          JUMPOUT(0x14A1DE0);
        case 47:
          JUMPOUT(0x14A1D20);
        default:
          JUMPOUT(0x14A1E38);
      }
    }
    v6 = 1;
    v5 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v4 + 816LL);
    if ( (char *)v5 != (char *)sub_1D5A400 )
      return ((unsigned __int8 (__fastcall *)(_QWORD, __int64))v5)(*(_QWORD *)(a1 + 24), a4) ^ 1u;
    return v6;
  }
  v5 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v4 + 784LL);
  if ( v5 == sub_2165A80 )
  {
    v6 = 1;
    if ( *(_BYTE *)(a4 + 8) == 11 && *(_BYTE *)(a3 + 8) == 11 && (unsigned int)sub_1643030(a4) == 64 )
      return (unsigned int)sub_1643030(a3) != 32;
    return v6;
  }
  return ((unsigned __int8 (__fastcall *)(_QWORD, __int64))v5)(*(_QWORD *)(a1 + 24), a4) ^ 1u;
}
