// Function: sub_3154660
// Address: 0x3154660
//
__int64 __fastcall sub_3154660(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12
  unsigned int v4; // r15d
  __int64 v5; // rsi
  _QWORD v7[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = a2 + 8;
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  v3 = *(_QWORD *)(a2 + 24);
  v7[0] = 0;
  if ( v3 != a2 + 8 )
  {
    v4 = 0;
    do
    {
      v5 = v4++;
      (*(void (__fastcall **)(__int64, __int64, _QWORD *))(*(_QWORD *)a1 + 32LL))(a1, v5, v7);
      sub_3154400(a1, *(_QWORD *)(v3 + 56), v3 + 64, (_QWORD *)(v3 + 208));
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 40LL))(a1, 0);
      v3 = sub_220EF30(v3);
    }
    while ( v2 != v3 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
