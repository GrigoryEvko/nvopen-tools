// Function: sub_828BC0
// Address: 0x828bc0
//
__int64 __fastcall sub_828BC0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // r12
  __int64 result; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1;
  *a2 = 0;
  result = sub_8D3C40(a1);
  if ( (_DWORD)result )
  {
    while ( *(_BYTE *)(v2 + 140) == 12 )
      v2 = *(_QWORD *)(v2 + 160);
    if ( !(unsigned int)sub_89EDD0(v2, unk_4D04988, v4) )
      sub_721090();
    *a2 = *(_QWORD *)(v4[0] + 32LL);
    return 1;
  }
  return result;
}
