// Function: sub_3215DE0
// Address: 0x3215de0
//
__int64 __fastcall sub_3215DE0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdi
  char v3; // al
  __int64 result; // rax
  __int16 v5; // dx
  __int64 v6; // rdi
  __int16 v7; // dx
  unsigned __int16 v8; // dx
  unsigned __int64 *v9; // rdi

  switch ( *(_DWORD *)a1 )
  {
    case 1:
      return sub_3215240((__int64 *)(a1 + 8), (unsigned int *)a2, *(_WORD *)(a1 + 6));
    case 2:
      return sub_32157B0((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
    case 3:
      return sub_3215440(a1 + 8, a2, *(_WORD *)(a1 + 6));
    case 4:
      return sub_3215500(a1 + 8, a2, *(_WORD *)(a1 + 6));
    case 5:
      return 4;
    case 6:
      return sub_3215620(*(_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
    case 7:
      return sub_32159A0(a1 + 8, a2, *(_WORD *)(a1 + 6));
    case 8:
      v5 = *(_WORD *)(a1 + 6);
      v6 = *(_QWORD *)(a1 + 8);
      switch ( v5 )
      {
        case 3:
          return (unsigned int)(*(_DWORD *)(v6 + 8) + 2);
        case 4:
          return (unsigned int)(*(_DWORD *)(v6 + 8) + 4);
        case 9:
        case 24:
          goto LABEL_19;
        case 10:
          return (unsigned int)(*(_DWORD *)(v6 + 8) + 1);
        case 30:
          return 16;
        default:
          BUG();
      }
    case 9:
      v7 = *(_WORD *)(a1 + 6);
      v6 = *(_QWORD *)(a1 + 8);
      switch ( v7 )
      {
        case 3:
          result = (unsigned int)(*(_DWORD *)(v6 + 8) + 2);
          break;
        case 4:
          result = (unsigned int)(*(_DWORD *)(v6 + 8) + 4);
          break;
        case 9:
        case 24:
LABEL_19:
          v2 = *(unsigned int *)(v6 + 8);
          result = (unsigned int)v2 + (unsigned int)sub_F03EF0(v2);
          break;
        case 10:
          result = (unsigned int)(*(_DWORD *)(v6 + 8) + 1);
          break;
        default:
          BUG();
      }
      return result;
    case 0xA:
      v8 = *(_WORD *)(a1 + 6);
      v9 = (unsigned __int64 *)(a1 + 8);
      if ( v8 != 23 )
      {
        if ( v8 > 0x17u )
        {
          if ( v8 == 34 )
            return sub_F03EF0(*v9);
        }
        else
        {
          if ( v8 == 6 )
            return 4;
          if ( v8 == 7 )
            return 8;
        }
LABEL_14:
        BUG();
      }
      v3 = *(_BYTE *)(a2 + 3);
      if ( !v3 )
        return 4;
      if ( v3 != 1 )
        goto LABEL_14;
      return 8;
    case 0xB:
      return (unsigned int)*(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL) + 1;
    case 0xC:
      return sub_3215DA0(*(__int64 **)(a1 + 8), (unsigned int *)a2);
    default:
      BUG();
  }
}
