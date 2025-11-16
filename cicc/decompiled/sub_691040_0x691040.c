// Function: sub_691040
// Address: 0x691040
//
__int64 __fastcall sub_691040(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdi
  char v7; // al
  __int64 result; // rax
  __int64 v9; // rdx
  _BOOL4 v10; // eax
  __int64 v11; // rax
  unsigned int v12[5]; // [rsp+1Ch] [rbp-14h] BYREF

  v5 = *(_QWORD *)(a1 + 40);
  v6 = *(_QWORD *)(a1 + 24);
  v12[0] = 0;
  v7 = *(_BYTE *)(v6 + 80);
  if ( v7 == 16 )
  {
    v6 = **(_QWORD **)(v6 + 88);
    v7 = *(_BYTE *)(v6 + 80);
  }
  if ( v7 == 24 )
  {
    v6 = *(_QWORD *)(v6 + 88);
    v7 = *(_BYTE *)(v6 + 80);
  }
  switch ( v7 )
  {
    case 20:
      if ( a3 )
      {
LABEL_17:
        v6 = 0;
        LODWORD(v9) = 0;
        break;
      }
      v9 = **(_QWORD **)(*(_QWORD *)(v6 + 88) + 328LL);
      break;
    case 21:
      v9 = **(_QWORD **)(*(_QWORD *)(v6 + 88) + 232LL);
      break;
    case 19:
      v11 = *(_QWORD *)(v6 + 88);
      LODWORD(v9) = 0;
      if ( (*(_BYTE *)(v11 + 160) & 2) == 0 )
        v9 = **(_QWORD **)(v11 + 32);
      break;
    default:
      switch ( v7 )
      {
        case 17:
          v10 = sub_8780F0() == 0;
          v12[0] = v10;
          break;
        case 2:
          v10 = *(_BYTE *)(*(_QWORD *)(v6 + 88) + 173LL) != 12;
          v12[0] = v10;
          break;
        case 13:
          goto LABEL_17;
        default:
          v12[0] = 1;
LABEL_12:
          *(_BYTE *)(a1 + 17) |= 0x20u;
          result = sub_6E50A0(v6, v5);
          *(_BYTE *)(a2 + 56) = 1;
          return result;
      }
      if ( v10 )
      {
        *(_BYTE *)(a1 + 17) |= 0x20u;
        result = sub_6E50A0(v6, v5);
        *(_BYTE *)(a2 + 56) = 1;
        return result;
      }
      goto LABEL_17;
  }
  *(_QWORD *)(a1 + 40) = sub_690FF0(
                           v6,
                           v5,
                           v9,
                           *(_QWORD *)(a2 + 24),
                           *(_QWORD *)(a2 + 32),
                           a1 + 8,
                           *(_DWORD *)(a2 + 40),
                           (__int64)v12,
                           *(_QWORD *)(a2 + 48));
  result = v12[0];
  if ( v12[0] )
    goto LABEL_12;
  return result;
}
