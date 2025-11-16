// Function: sub_393D000
// Address: 0x393d000
//
__int64 *__fastcall sub_393D000(__int64 *a1, __int64 a2, int a3)
{
  switch ( a3 )
  {
    case 0:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Success", (__int64)"");
      break;
    case 1:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Invalid sample profile data (bad magic)", (__int64)"");
      break;
    case 2:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Unsupported sample profile format version", (__int64)"");
      break;
    case 3:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, &byte_4530795[-21], (__int64)byte_4530795);
      break;
    case 4:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, &byte_45307AC[-22], (__int64)byte_45307AC);
      break;
    case 5:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, &byte_45307CA[-29], (__int64)byte_45307CA);
      break;
    case 6:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Unrecognized sample profile encoding format", (__int64)"");
      break;
    case 7:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Profile encoding format unsupported for writing operations", (__int64)"");
      break;
    case 8:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, &byte_45307E8[-29], (__int64)byte_45307E8);
      break;
    case 9:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, &aFeature[-14], (__int64)"");
      break;
    case 10:
      *a1 = (__int64)(a1 + 2);
      sub_393CF10(a1, "Counter overflow", (__int64)"");
      break;
  }
  return a1;
}
