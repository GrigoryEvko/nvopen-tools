// Function: sub_16D7E20
// Address: 0x16d7e20
//
char __fastcall sub_16D7E20(__int64 *a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 v11; // rax
  char result; // al

  *a1 = (__int64)(a1 + 2);
  sub_16D6060(a1, a2, (__int64)&a2[a3]);
  a1[4] = (__int64)(a1 + 6);
  sub_16D6060(a1 + 4, a4, (__int64)&a4[a5]);
  a1[8] = 0;
  a1[9] = 0;
  a1[10] = 0;
  a1[11] = 0;
  if ( !qword_4FA1610 )
    sub_16C1EA0((__int64)&qword_4FA1610, sub_160CFB0, (__int64)sub_160D0B0, v7, v8, v9);
  v10 = qword_4FA1610;
  if ( (unsigned __int8)sub_16D5D40() )
    sub_16C30C0((pthread_mutex_t **)v10);
  else
    ++*(_DWORD *)(v10 + 8);
  v11 = qword_4FA13F0;
  if ( qword_4FA13F0 )
    *(_QWORD *)(qword_4FA13F0 + 96) = a1 + 13;
  a1[13] = v11;
  a1[12] = (__int64)&qword_4FA13F0;
  qword_4FA13F0 = (__int64)a1;
  result = sub_16D5D40();
  if ( result )
    return sub_16C30E0((pthread_mutex_t **)v10);
  --*(_DWORD *)(v10 + 8);
  return result;
}
