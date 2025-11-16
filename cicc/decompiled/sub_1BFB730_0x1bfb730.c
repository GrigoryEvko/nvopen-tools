// Function: sub_1BFB730
// Address: 0x1bfb730
//
__int64 __fastcall sub_1BFB730(__int64 a1, unsigned int a2, char a3)
{
  __int64 result; // rax
  int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  result = 32764;
  if ( a2 <= 0x45 )
  {
    v4 = -(a3 == 0);
    LOBYTE(v4) = 0;
    result = (unsigned int)(v4 + 4352);
  }
  *(_DWORD *)(a1 + 8) = a2;
  *(_DWORD *)a1 = result;
  *(_DWORD *)(a1 + 4) = 0;
  *(_QWORD *)(a1 + 12) = 0;
  *(_QWORD *)(a1 + 20) = 0;
  *(_QWORD *)(a1 + 28) = 0;
  *(_QWORD *)(a1 + 36) = 0;
  *(_QWORD *)(a1 + 44) = 0;
  switch ( a2 )
  {
    case 0x14u:
    case 0x15u:
      *(_QWORD *)(a1 + 12) = 0x4000008000LL;
      *(_QWORD *)(a1 + 20) = 0x3F00000002LL;
      *(_QWORD *)(a1 + 28) = 0x800000014LL;
      *(_QWORD *)(a1 + 36) = 0x3000000001LL;
      *(_QWORD *)(a1 + 44) = 0x200000020LL;
      return 0x200000020LL;
    case 0x1Eu:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      v7 = 0x3F00000008LL;
      goto LABEL_10;
    case 0x20u:
    case 0x23u:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      v7 = 0xFF00000008LL;
LABEL_10:
      *(_QWORD *)(a1 + 20) = v7;
      v6 = 0x1000000020LL;
      goto LABEL_7;
    case 0x25u:
      *(_QWORD *)(a1 + 12) = 0x10000020000LL;
      *(_QWORD *)(a1 + 20) = 0xFF00000008LL;
      *(_QWORD *)(a1 + 28) = 0x2000000020LL;
      *(_QWORD *)(a1 + 36) = 0x8000000002LL;
      *(_QWORD *)(a1 + 44) = 0x400000020LL;
      return 0x400000020LL;
    case 0x32u:
    case 0x34u:
    case 0x35u:
    case 0x3Cu:
    case 0x3Du:
    case 0x3Eu:
    case 0x46u:
    case 0x48u:
    case 0x50u:
    case 0x52u:
    case 0x57u:
    case 0x5Au:
    case 0x64u:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      *(_QWORD *)(a1 + 20) = 0xFF00000008LL;
      v6 = 0x2000000020LL;
LABEL_7:
      *(_QWORD *)(a1 + 28) = v6;
      *(_QWORD *)(a1 + 36) = 0x4000000001LL;
      *(_QWORD *)(a1 + 44) = 0x400000020LL;
      return 0x400000020LL;
    case 0x49u:
    case 0x4Bu:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      *(_QWORD *)(a1 + 20) = 0xFF00000008LL;
      *(_QWORD *)(a1 + 28) = 0x1000000020LL;
      *(_QWORD *)(a1 + 36) = 0x2000000001LL;
      *(_QWORD *)(a1 + 44) = 0x400000020LL;
      return 0x400000020LL;
    case 0x56u:
    case 0x58u:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      *(_QWORD *)(a1 + 20) = 0xFF00000008LL;
      v5 = 0x1000000020LL;
      goto LABEL_5;
    case 0x59u:
    case 0x65u:
    case 0x68u:
      *(_QWORD *)(a1 + 12) = 0x10000010000LL;
      *(_QWORD *)(a1 + 20) = 0xFF00000008LL;
      v5 = 0x1800000020LL;
LABEL_5:
      *(_QWORD *)(a1 + 28) = v5;
      *(_QWORD *)(a1 + 36) = 0x3000000001LL;
      *(_QWORD *)(a1 + 44) = 0x400000020LL;
      result = 0x400000020LL;
      break;
    default:
      return result;
  }
  return result;
}
