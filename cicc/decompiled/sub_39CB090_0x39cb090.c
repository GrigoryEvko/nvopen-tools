// Function: sub_39CB090
// Address: 0x39cb090
//
__int64 __fastcall sub_39CB090(__int64 *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 **v7; // r14
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  int v11; // eax
  char v12[4]; // [rsp+8h] [rbp-28h] BYREF
  int v13; // [rsp+Ch] [rbp-24h]

  v2 = sub_39C84F0(a1);
  v3 = sub_39A8220((__int64)a1, a2, v2);
  sub_39CB010(a1, v3, *(_QWORD *)(a1[24] + 384), *(_QWORD *)(a1[24] + 392));
  v4 = a1[25];
  if ( *(_BYTE *)(v4 + 4512) && !(unsigned __int8)sub_39DF900(*(_QWORD *)(*(_QWORD *)(v4 + 4008) + 8LL) + 792LL) )
    sub_39A34D0((__int64)a1, v3, 16359);
  if ( !sub_39C84F0(a1) )
  {
    v5 = *(_QWORD *)(a1[24] + 264);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(v5 + 8) + 504LL) - 34) > 1 )
    {
      v9 = *(__int64 (**)())(**(_QWORD **)(v5 + 16) + 112LL);
      if ( v9 == sub_1D00B10 )
        BUG();
      v10 = v9();
      v11 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v10 + 416LL))(v10, *(_QWORD *)(a1[24] + 264));
      v12[0] = 1;
      v13 = v11;
      if ( v11 > 0 )
        sub_39C9790(a1, v3, 64, (__int64)v12);
    }
    else
    {
      v6 = sub_145CDC0(0x10u, a1 + 11);
      v7 = (__int64 **)v6;
      if ( v6 )
      {
        *(_QWORD *)v6 = 0;
        *(_DWORD *)(v6 + 8) = 0;
      }
      sub_39A35E0((__int64)a1, (__int64 *)v6, 11, 156);
      sub_39A4520(a1, v3, 64, v7);
    }
  }
  sub_398FCD0(a1[25], a2, v3);
  return v3;
}
