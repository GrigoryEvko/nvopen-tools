// Function: sub_E97D30
// Address: 0xe97d30
//
void (*__fastcall sub_E97D30(__int64 *a1, unsigned __int64 a2, unsigned int a3))()
{
  unsigned __int64 *v3; // r8
  __int64 v4; // rax
  void (*result)(); // rax
  unsigned __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = &v6;
  if ( !*(_BYTE *)(*(_QWORD *)(a1[1] + 152) + 16LL) )
  {
    a2 = _byteswap_uint64(a2);
    v3 = (unsigned __int64 *)((char *)&v6 + 8 - a3);
  }
  v4 = *a1;
  v6 = a2;
  result = *(void (**)())(v4 + 512);
  if ( result != nullsub_360 )
    return (void (*)())((__int64 (__fastcall *)(__int64 *, unsigned __int64 *, _QWORD))result)(a1, v3, a3);
  return result;
}
