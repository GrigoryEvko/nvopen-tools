// Function: sub_865900
// Address: 0x865900
//
__int64 __fastcall sub_865900(__int64 a1)
{
  __int64 v1; // rbx
  int v2; // r13d
  char v3; // dl
  __int64 v4; // r14
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rax

  v1 = a1;
  v2 = unk_4F04C2C;
  if ( !dword_4F07734 )
  {
    if ( a1 && (unsigned __int8)(*(_BYTE *)(a1 + 80) - 19) >= 2u )
      v1 = 0;
    goto LABEL_7;
  }
  if ( !a1 )
  {
LABEL_7:
    v4 = 0;
    v5 = sub_878CA0();
    goto LABEL_8;
  }
  v3 = *(_BYTE *)(a1 + 80);
  switch ( v3 )
  {
    case 4:
    case 5:
      v4 = 0;
      v8 = *(_QWORD **)(*(_QWORD *)(a1 + 96) + 80LL);
      goto LABEL_20;
    case 6:
      v8 = *(_QWORD **)(*(_QWORD *)(a1 + 96) + 32LL);
      goto LABEL_25;
    case 9:
      v8 = *(_QWORD **)(*(_QWORD *)(a1 + 96) + 56LL);
      goto LABEL_25;
    case 10:
      v8 = *(_QWORD **)(*(_QWORD *)(a1 + 96) + 56LL);
      goto LABEL_19;
    case 19:
    case 20:
    case 21:
    case 22:
      v8 = *(_QWORD **)(a1 + 88);
      goto LABEL_18;
    default:
      v8 = 0;
LABEL_18:
      if ( v3 == 20 || v3 == 10 )
      {
LABEL_19:
        v5 = v8[41];
        v4 = v8[22];
        if ( v5 )
          break;
      }
      else
      {
LABEL_25:
        v4 = 0;
      }
LABEL_20:
      v5 = v8[4];
      break;
  }
LABEL_8:
  sub_864700(v5, 0, v4, 0, v1, 0, 0, 0x1004u);
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  *(_BYTE *)(result + 8) |= 0x20u;
  if ( dword_4F07734 )
  {
    result = dword_4F07730;
    if ( !dword_4F07730 )
    {
      if ( v1 )
      {
        sub_8600D0(0xEu, -1, 0, v4);
        result = qword_4F04C68[0] + 776LL * dword_4F04C64;
        *(_QWORD *)(result + 368) = v1;
        if ( (*(_BYTE *)(v1 + 85) & 1) != 0 )
          ++*(_DWORD *)(result + 200);
      }
    }
  }
  if ( v2 != -1 )
  {
    v7 = qword_4F04C68[0];
    *(_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 452) = v2;
    result = (__int64)&dword_4F077C4;
    if ( dword_4F077C4 == 2 )
    {
      result = 776LL * (int)dword_4F04C40;
      *(_BYTE *)(v7 + result + 7) |= 8u;
    }
  }
  return result;
}
