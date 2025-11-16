// Function: sub_C53130
// Address: 0xc53130
//
unsigned __int64 __fastcall sub_C53130(__int64 a1)
{
  __int64 v1; // r12
  unsigned __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-28h] BYREF
  _QWORD v4[4]; // [rsp+10h] [rbp-20h] BYREF

  if ( !qword_4F83CE0 )
    sub_C7D570(&qword_4F83CE0, sub_C53DA0, sub_C50EC0);
  v1 = qword_4F83CE0;
  v3 = a1;
  if ( (*(_BYTE *)(a1 + 13) & 0x20) != 0 )
  {
    result = *(unsigned int *)(qword_4F83CE0 + 80);
    if ( result + 1 > *(unsigned int *)(qword_4F83CE0 + 84) )
    {
      sub_C8D5F0(qword_4F83CE0 + 72, qword_4F83CE0 + 88, result + 1, 8);
      result = *(unsigned int *)(v1 + 80);
    }
    *(_QWORD *)(*(_QWORD *)(v1 + 72) + 8 * result) = a1;
    ++*(_DWORD *)(v1 + 80);
    *(_BYTE *)(a1 + 13) |= 0x40u;
  }
  else
  {
    v4[1] = qword_4F83CE0;
    v4[0] = &v3;
    result = (unsigned __int64)sub_C52DD0(
                                 qword_4F83CE0,
                                 a1,
                                 (__int64 (__fastcall *)(__int64, __int64))sub_C53EC0,
                                 (__int64)v4);
    *(_BYTE *)(a1 + 13) |= 0x40u;
  }
  return result;
}
