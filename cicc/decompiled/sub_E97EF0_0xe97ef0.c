// Function: sub_E97EF0
// Address: 0xe97ef0
//
void (*__fastcall sub_E97EF0(__int64 *a1, __int64 a2, __int64 a3, int a4))()
{
  __int64 v4; // rax
  void (*result)(); // rax
  __int16 v6; // [rsp+28h] [rbp-28h] BYREF
  int v7; // [rsp+2Ah] [rbp-26h]

  v4 = *a1;
  v7 = a4;
  result = *(void (**)())(v4 + 760);
  v6 = 4418;
  if ( result != nullsub_348 )
    return (void (*)())((__int64 (__fastcall *)(__int64 *, __int64, __int64, __int16 *, __int64))result)(
                         a1,
                         a2,
                         a3,
                         &v6,
                         6);
  return result;
}
