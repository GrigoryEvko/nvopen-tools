// Function: sub_2BDEB70
// Address: 0x2bdeb70
//
__int64 __fastcall sub_2BDEB70(_QWORD *a1)
{
  __int64 v1; // r12
  __int64 *v3; // rdi
  const char *v4; // rsi
  int v5; // eax
  __int64 v6; // rdx
  char *v7; // rsi
  __int64 result; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v10)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v11)(__int64, void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD), _QWORD *, _QWORD *); // [rsp+18h] [rbp-28h]

  v1 = (__int64)(a1 + 4);
  v3 = a1 + 5;
  *(v3 - 5) = 0;
  *(v3 - 4) = 0;
  *(v3 - 3) = 0;
  *(v3 - 1) = (__int64)&unk_4A23850;
  *(v3 - 2) = 0;
  a1[5] = a1 + 7;
  sub_2BDC240(v3, "fpm", (__int64)"");
  v4 = s;
  a1[9] = a1 + 11;
  a1[10] = 0x600000000LL;
  a1[4] = &unk_4A34660;
  v5 = sub_2241AC0((__int64)&qword_5010808, v4);
  v6 = 58;
  v7 = "seed-collection<tr-save,bottom-up-vec,tr-accept-or-revert>";
  v9[0] = sub_2BEEFF0;
  v11 = sub_2BDB650;
  v10 = sub_2BDB680;
  if ( v5 )
  {
    v7 = (char *)qword_5010808;
    v6 = qword_5010810;
  }
  sub_2BDE770(v1, v7, v6, (__int64)v9);
  result = (__int64)v10;
  if ( v10 )
    return v10(v9, v9, 3);
  return result;
}
