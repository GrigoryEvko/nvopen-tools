// Function: sub_1643030
// Address: 0x1643030
//
__int64 __fastcall sub_1643030(__int64 a1)
{
  int v1; // eax
  __int64 result; // rax
  __int64 v3; // rdx

  v1 = 1;
  while ( 2 )
  {
    switch ( *(_BYTE *)(a1 + 8) )
    {
      case 1:
        goto LABEL_7;
      case 2:
        return (unsigned int)(32 * v1);
      case 3:
      case 9:
        return (unsigned int)(v1 << 6);
      case 4:
        v1 *= 5;
LABEL_7:
        result = (unsigned int)(16 * v1);
        break;
      case 5:
      case 6:
        result = (unsigned int)(v1 << 7);
        break;
      case 0xB:
        result = (unsigned int)((*(_DWORD *)(a1 + 8) >> 8) * v1);
        break;
      case 0x10:
        v3 = *(_QWORD *)(a1 + 32);
        a1 = *(_QWORD *)(a1 + 24);
        v1 *= (_DWORD)v3;
        continue;
      default:
        result = 0;
        break;
    }
    return result;
  }
}
