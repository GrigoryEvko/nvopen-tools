// Function: sub_2618FC0
// Address: 0x2618fc0
//
__int64 __fastcall sub_2618FC0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax

  switch ( *(_DWORD *)(a1 + 48) )
  {
    case 1:
      goto LABEL_5;
    case 3:
      goto LABEL_3;
    case 0xE:
    case 0x1C:
    case 0x1D:
      goto LABEL_4;
    case 0x24:
      if ( !*(_BYTE *)(a1 + 41) )
        goto LABEL_11;
LABEL_3:
      if ( sub_2618C40(a1) )
        goto LABEL_4;
LABEL_5:
      result = 4;
      break;
    case 0x26:
    case 0x27:
      v2 = sub_BA91D0(*(_QWORD *)a1, "cf-protection-branch", 0x14u);
      if ( v2
        && (v3 = *(_QWORD *)(v2 + 136)) != 0
        && (*(_DWORD *)(v3 + 32) <= 0x40u ? (v4 = *(_QWORD *)(v3 + 24)) : (v4 = **(_QWORD **)(v3 + 24)), v4) )
      {
LABEL_11:
        result = 16;
      }
      else
      {
LABEL_4:
        result = 8;
      }
      break;
    default:
      sub_C64ED0("Unsupported architecture for jump tables", 1u);
  }
  return result;
}
