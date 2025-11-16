// Function: sub_39B2770
// Address: 0x39b2770
//
__int64 __fastcall sub_39B2770(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  __int64 (*v5)(); // r9
  __int64 result; // rax
  __int64 savedregs; // [rsp+0h] [rbp+0h] BYREF

  v4 = *(_QWORD *)(a1 + 24);
  if ( a2 == 36 )
  {
    v5 = *(__int64 (**)())(*(_QWORD *)v4 + 784LL);
    result = 1;
    if ( v5 != sub_1D5A3F0 )
      return ((unsigned __int8 (__fastcall *)(_QWORD, __int64))v5)(*(_QWORD *)(a1 + 24), a4) ^ 1u;
  }
  else
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
    v5 = *(__int64 (**)())(*(_QWORD *)v4 + 816LL);
    result = 1;
    if ( v5 != sub_1D5A400 )
      return ((unsigned __int8 (__fastcall *)(_QWORD, __int64))v5)(*(_QWORD *)(a1 + 24), a4) ^ 1u;
  }
  return result;
}
