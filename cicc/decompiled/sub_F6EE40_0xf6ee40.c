// Function: sub_F6EE40
// Address: 0xf6ee40
//
bool __fastcall sub_F6EE40(__int64 *a1, __int64 a2)
{
  bool result; // al
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // r12

  result = 1;
  v3 = *a1;
  if ( *a1 )
  {
    v4 = sub_D47930((__int64)a1);
    v5 = sub_DBA6E0(a2, (__int64)a1, v4, 0);
    return !sub_D96A50(v5) && *(_BYTE *)(sub_D95540(v5) + 8) == 12 && (unsigned int)sub_DAD860(a2, v5, v3) == 1;
  }
  return result;
}
