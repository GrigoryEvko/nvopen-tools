// Function: sub_C30940
// Address: 0xc30940
//
__int64 __fastcall sub_C30940(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // rsi
  __m128i *v7; // [rsp+8h] [rbp-48h]
  _QWORD v8[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (*(unsigned int (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 16))(a1) )
    v3 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v3 )
  {
    v4 = 0;
    do
    {
      while ( !(*(unsigned __int8 (__fastcall **)(__int64 *, _QWORD, _QWORD *))(*a1 + 32))(a1, (unsigned int)v4, v8) )
      {
        if ( ++v4 == v3 )
          return (*(__int64 (__fastcall **)(__int64 *))(*a1 + 48))(a1);
      }
      v5 = v4++;
      v7 = (__m128i *)(*(_QWORD *)a2 + (v5 << 6));
      (*(void (__fastcall **)(__int64 *))(*a1 + 104))(a1);
      sub_C30620(a1, v7);
      (*(void (__fastcall **)(__int64 *))(*a1 + 112))(a1);
      (*(void (__fastcall **)(__int64 *, _QWORD))(*a1 + 40))(a1, v8[0]);
    }
    while ( v4 != v3 );
  }
  return (*(__int64 (__fastcall **)(__int64 *))(*a1 + 48))(a1);
}
