// Function: sub_266EEF0
// Address: 0x266eef0
//
__int64 __fastcall sub_266EEF0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v3; // rax
  __int64 v4; // rax

  v1 = sub_B2BE50(a1);
  if ( sub_B6EA50(v1) )
    return 1;
  v3 = sub_B2BE50(a1);
  v4 = sub_B6F970(v3);
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 48LL))(v4);
}
