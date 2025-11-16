// Function: sub_C37950
// Address: 0xc37950
//
__int64 __fastcall sub_C37950(__int64 a1, __int64 a2)
{
  char *v3; // rax
  char v4; // al
  unsigned __int8 v5; // cl
  char v6; // si
  __int64 result; // rax
  unsigned int v8; // [rsp+Ch] [rbp-64h]
  _QWORD v9[4]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v10[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = (char *)sub_C94E20(qword_4F863F0);
  if ( v3 )
    v4 = *v3;
  else
    v4 = qword_4F863F0[2];
  if ( v4 && *(_UNKNOWN **)a1 == &unk_3F657C0 && (sub_C33940(a1) || sub_C33940(a2)) )
  {
    sub_C33EB0(v9, (__int64 *)a1);
    sub_C33EB0(v10, (__int64 *)a2);
    if ( sub_C33940((__int64)v9) )
      sub_C37310((__int64)v9, 0);
    if ( sub_C33940((__int64)v10) )
      sub_C37310((__int64)v10, 0);
    v8 = sub_C37950(v9, v10);
    sub_C338F0((__int64)v10);
    sub_C338F0((__int64)v9);
    return v8;
  }
  v5 = *(_BYTE *)(a1 + 20);
  v6 = *(_BYTE *)(a2 + 20);
  switch ( (*(_BYTE *)(a2 + 20) & 7) + 4 * (v5 & 7) )
  {
    case 0:
      result = 1;
      if ( ((v5 ^ *(_BYTE *)(a2 + 20)) & 8) != 0 )
        goto LABEL_9;
      return result;
    case 1:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 0xD:
      return 3;
    case 2:
    case 3:
    case 0xB:
      goto LABEL_9;
    case 8:
    case 0xC:
    case 0xE:
      if ( (v6 & 8) != 0 )
        return 2;
      return 0;
    case 0xA:
      if ( ((v5 ^ *(_BYTE *)(a2 + 20)) & 8) != 0 )
      {
LABEL_9:
        if ( (*(_BYTE *)(a1 + 20) & 8) != 0 )
          return 0;
        return 2;
      }
      result = sub_C37580(a1, a2);
      if ( (*(_BYTE *)(a1 + 20) & 8) == 0 )
        return result;
      if ( !(_DWORD)result )
        return 2;
      if ( (_DWORD)result == 2 )
        return 0;
      return result;
    case 0xF:
      return 1;
    default:
      BUG();
  }
}
