// Function: sub_AD6530
// Address: 0xad6530
//
__int64 __fastcall sub_AD6530(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 result; // rax
  __int64 v10[8]; // [rsp+0h] [rbp-40h] BYREF

  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
      v2 = sub_BCAC60(a1);
      v6 = sub_C33340(a1, a2, v3, v4, v5);
      v7 = v6;
      if ( v2 == v6 )
        sub_C3C500(v10, v6, 0);
      else
        sub_C373C0(v10, v2, 0);
      if ( v10[0] == v7 )
        sub_C3CEB0(v10, 0);
      else
        sub_C37310(v10, 0);
      v8 = sub_AC8EA0(*(__int64 **)a1, v10);
      sub_91D830(v10);
      result = v8;
      break;
    case 0xB:
      result = sub_AC3540(*(__int64 **)a1);
      break;
    case 0xC:
      result = sub_AD64C0(a1, 0, 0);
      break;
    case 0xE:
      result = sub_AC9EC0((__int64 **)a1);
      break;
    case 0xF:
    case 0x10:
    case 0x11:
    case 0x12:
      result = sub_AC9350((__int64 **)a1);
      break;
    case 0x14:
      result = sub_ACA3B0((__int64 **)a1);
      break;
    default:
      BUG();
  }
  return result;
}
