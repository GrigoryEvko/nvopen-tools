// Function: sub_2AAF930
// Address: 0x2aaf930
//
__int64 __fastcall sub_2AAF930(__int64 a1, unsigned __int8 *a2)
{
  int v2; // esi
  int v3; // edx
  __int64 v4; // rax
  __int64 v5; // kr08_8
  __int64 result; // rax
  int v8; // ecx
  int v9; // edx

  v5 = v4;
  result = *(unsigned __int8 *)(a1 + 152);
  switch ( *(_BYTE *)(a1 + 152) )
  {
    case 0:
      return result;
    case 1:
      sub_B447F0(a2, *(_BYTE *)(a1 + 156) & 1);
      result = sub_B44850(a2, (*(_BYTE *)(a1 + 156) & 2) != 0);
      break;
    case 2:
      v8 = *(_BYTE *)(a1 + 156) & 1;
      v9 = a2[1] & 1;
      result = v9 | (2 * (v8 | (a2[1] >> 1) & 0xFEu));
      a2[1] = v9 | (2 * (v8 | (a2[1] >> 1) & 0xFE));
      break;
    case 3:
      result = sub_B448B0((__int64)a2, *(_BYTE *)(a1 + 156) & 1);
      break;
    case 4:
      result = sub_B4DDE0((__int64)a2, *(_DWORD *)(a1 + 156));
      break;
    case 5:
      sub_B44ED0((__int64)a2, *(_BYTE *)(a1 + 156) & 1);
      sub_B44EF0((__int64)a2, (*(_BYTE *)(a1 + 156) & 2) != 0);
      sub_B44F10((__int64)a2, (*(_BYTE *)(a1 + 156) & 4) != 0);
      sub_B450D0((__int64)a2, (*(_BYTE *)(a1 + 156) & 8) != 0);
      sub_B450F0((__int64)a2, (*(_BYTE *)(a1 + 156) & 0x10) != 0);
      sub_B45110((__int64)a2, (*(_BYTE *)(a1 + 156) & 0x20) != 0);
      v2 = ((*(_BYTE *)(a1 + 156) & 0x40) != 0) << 6;
      v3 = a2[1] & 1;
      result = v3 | (2 * (v2 | (a2[1] >> 1) & 0xBFu));
      a2[1] = v3 | (2 * (v2 | (a2[1] >> 1) & 0xBF));
      break;
    case 6:
      result = sub_B448D0((__int64)a2, *(_BYTE *)(a1 + 156) & 1);
      break;
    default:
      result = v5;
      break;
  }
  return result;
}
