// Function: sub_2567770
// Address: 0x2567770
//
__int64 __fastcall sub_2567770(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  unsigned __int8 v7; // al
  unsigned __int8 v8; // [rsp+Fh] [rbp-41h]
  __m128i v9[4]; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 16) == sub_B43CB0(a2) )
  {
    v5 = **(_QWORD **)(a1 + 24);
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_QWORD *)(a1 + 40);
    v4 = sub_B43CB0(a2);
    sub_250D230((unsigned __int64 *)v9, v4, 4, 0);
    v5 = sub_2567630(v2, v9, v3, 2, 0);
  }
  if ( !v5 )
    return 0;
  if ( !**(_BYTE **)(a1 + 48)
    && (!**(_BYTE **)(a1 + 56)
     || !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v5 + 120LL))(
           v5,
           *(_QWORD *)(a1 + 32),
           a2)) )
  {
    if ( **(_BYTE **)(a1 + 64) )
    {
      v7 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v5 + 112LL))(v5, *(_QWORD *)(a2 + 40));
      if ( v7 )
      {
        v8 = v7;
        sub_250ED80(*(_QWORD *)(a1 + 32), v5, *(_QWORD *)(a1 + 40), 1);
        return v8;
      }
    }
    return 0;
  }
  sub_250ED80(*(_QWORD *)(a1 + 32), v5, *(_QWORD *)(a1 + 40), 1);
  return 1;
}
