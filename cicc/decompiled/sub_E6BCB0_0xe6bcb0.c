// Function: sub_E6BCB0
// Address: 0xe6bcb0
//
__int64 __fastcall sub_E6BCB0(_DWORD *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rdx
  __int64 v8; // rdx
  _BOOL8 v9; // rdx

  switch ( *a1 )
  {
    case 0:
      result = sub_EA1530(32, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)result = 0;
      *(_QWORD *)(result + 24) = 0;
      v7 = a2 != 0;
      v6 = 2LL * a3;
      BYTE1(v7) = 4;
      goto LABEL_10;
    case 1:
      result = sub_EA1530(40, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      v8 = a2 != 0;
      *(_QWORD *)result = 0;
      BYTE1(v8) = 2;
      *(_DWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 8) = *(_QWORD *)(result + 8) & 0xFFFF0000FFF00000LL | (2LL * a3) | v8;
      if ( a2 )
        *(_QWORD *)(result - 8) = a2;
      *(_QWORD *)(result + 32) = 0;
      return result;
    case 2:
      result = sub_EA1530(32, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      v7 = a2 != 0;
      *(_QWORD *)result = 0;
      v6 = 2LL * a3;
      BYTE1(v7) = 3;
      goto LABEL_10;
    case 3:
      result = sub_EA1530(40, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      v9 = a2 != 0;
      *(_QWORD *)result = 0;
      BYTE1(v9) = 1;
      *(_DWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 8) = *(_QWORD *)(result + 8) & 0xFFFF0000FFF00000LL | (2LL * a3) | v9;
      if ( a2 )
        *(_QWORD *)(result - 8) = a2;
      *(_WORD *)(result + 32) = 0;
      return result;
    case 4:
      result = sub_EA1530(32, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      v7 = 2LL * a3;
      *(_QWORD *)result = 0;
      v6 = a2 != 0;
LABEL_10:
      *(_DWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 8) = *(_QWORD *)(result + 8) & 0xFFFF0000FFF00000LL | v6 | v7;
      if ( a2 )
        goto LABEL_11;
      return result;
    case 5:
      result = sub_EA1530(192, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      v5 = a2 != 0;
      *(_QWORD *)result = 0;
      BYTE1(v5) = 5;
      *(_DWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 8) = *(_QWORD *)(result + 8) & 0xFFFF0000FFF00000LL | (2LL * a3) | v5;
      if ( a2 )
        *(_QWORD *)(result - 8) = a2;
      *(_BYTE *)(result + 36) = 0;
      *(_DWORD *)(result + 40) = 0;
      *(_WORD *)(result + 44) = 0;
      *(_BYTE *)(result + 64) = 0;
      *(_BYTE *)(result + 88) = 0;
      *(_BYTE *)(result + 112) = 0;
      *(_QWORD *)(result + 120) = 0;
      *(_BYTE *)(result + 130) = 0;
      *(_BYTE *)(result + 176) = 0;
      *(_QWORD *)(result + 184) = 0;
      return result;
    case 6:
      return sub_E6B5A0((__int64)a1, a2, a3);
    default:
      result = sub_EA1530(32, a2, a1);
      if ( !result )
        return 0;
      *(_QWORD *)(result + 24) = 0;
      *(_QWORD *)result = 0;
      *(_DWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 8) = *(_QWORD *)(result + 8) & 0xFFFF0000FFF00000LL | (a2 != 0) | (2LL * a3);
      if ( a2 )
LABEL_11:
        *(_QWORD *)(result - 8) = a2;
      return result;
  }
}
