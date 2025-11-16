// Function: sub_254F400
// Address: 0x254f400
//
__int64 __fastcall sub_254F400(__int64 a1, _BYTE *a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 result; // rax

  if ( !sub_B2FC80((__int64)a2) && !(unsigned __int8)sub_B2FC00(a2) )
    return 1;
  result = sub_B19060(*(_QWORD *)(a1 + 208) + 248LL, (__int64)a2, v2, v3);
  if ( (_BYTE)result )
    return 1;
  if ( *(_QWORD *)(a1 + 4432) )
    return (*(__int64 (__fastcall **)(__int64, _BYTE *))(a1 + 4440))(a1 + 4416, a2);
  return result;
}
