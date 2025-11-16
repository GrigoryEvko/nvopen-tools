// Function: sub_890490
// Address: 0x890490
//
__int64 __fastcall sub_890490(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  int v3; // edx
  __int64 result; // rax

  v2 = *a2;
  v3 = *(unsigned __int8 *)(*a2 + 80);
  switch ( (char)v3 )
  {
    case 4:
    case 5:
      result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 80LL);
      goto LABEL_3;
    case 6:
      result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 32LL);
      goto LABEL_10;
    case 9:
    case 10:
      result = *(_QWORD *)(*(_QWORD *)(v2 + 96) + 56LL);
LABEL_10:
      if ( result )
        goto LABEL_11;
      goto LABEL_5;
    case 19:
    case 20:
    case 21:
    case 22:
      result = *(_QWORD *)(v2 + 88);
LABEL_3:
      if ( !result )
        goto LABEL_4;
LABEL_11:
      if ( (*(_BYTE *)(result + 160) & 1) != 0 )
        return result;
      goto LABEL_4;
    default:
      result = (unsigned int)(v3 - 4);
LABEL_4:
      if ( (unsigned __int8)(v3 - 4) > 1u )
        goto LABEL_5;
      result = *(unsigned __int8 *)(*(_QWORD *)(v2 + 88) + 177LL);
      if ( (result & 0x80u) != 0LL )
      {
        *(_QWORD *)(a1 + 64) = 0x100000001LL;
        return 0x100000001LL;
      }
      else if ( (result & 0x20) != 0 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 456) = v2;
      }
      else
      {
        *(_DWORD *)(a1 + 64) = 1;
      }
      return result;
  }
}
