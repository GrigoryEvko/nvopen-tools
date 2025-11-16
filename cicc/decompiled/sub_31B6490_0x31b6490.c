// Function: sub_31B6490
// Address: 0x31b6490
//
__int64 __fastcall sub_31B6490(_QWORD *a1, _BYTE *a2, size_t a3)
{
  __int64 *v4; // rdi
  __int64 (__fastcall *v5)(__int64, void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD), _QWORD *, _QWORD *); // rax
  __int64 result; // rax
  _QWORD v8[2]; // [rsp+10h] [rbp-90h] BYREF
  void (__fastcall *v9)(_BYTE *, _BYTE *, __int64); // [rsp+20h] [rbp-80h]
  __int64 (__fastcall *v10)(__int64, void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD), _QWORD *, _QWORD *); // [rsp+28h] [rbp-78h]
  _QWORD v11[2]; // [rsp+30h] [rbp-70h] BYREF
  void (__fastcall *v12)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-60h]
  __int64 (__fastcall *v13)(__int64, void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD), _QWORD *, _QWORD *); // [rsp+48h] [rbp-58h]
  _BYTE v14[16]; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v15)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-40h]
  __int64 (__fastcall *v16)(__int64, void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD), _QWORD *, _QWORD *); // [rsp+68h] [rbp-38h]

  v4 = a1 + 1;
  *(v4 - 1) = (__int64)&unk_4A23850;
  a1[1] = a1 + 3;
  sub_31B5580(v4, "regions-from-bbs", (__int64)"");
  v12 = 0;
  *a1 = &unk_4A348E0;
  v8[0] = sub_2BEEC60;
  v10 = sub_31B5520;
  v9 = (void (__fastcall *)(_BYTE *, _BYTE *, __int64))sub_31B5550;
  sub_31B5550(v11, v8, 2);
  v5 = v10;
  a1[5] = &unk_4A23850;
  v13 = v5;
  v12 = v9;
  a1[6] = a1 + 8;
  sub_31B5580(a1 + 6, "rpm", (__int64)"");
  v15 = 0;
  a1[5] = &unk_4A34630;
  a1[10] = a1 + 12;
  a1[11] = 0x600000000LL;
  if ( v12 )
  {
    v12(v14, v11, 2);
    v16 = v13;
    v15 = v12;
  }
  sub_31B5FA0((__int64)(a1 + 5), a2, a3, (__int64)v14);
  if ( v15 )
    v15(v14, v14, 3);
  if ( v12 )
    v12(v11, v11, 3);
  a1[5] = &unk_4A34690;
  result = (__int64)v9;
  if ( v9 )
    return ((__int64 (__fastcall *)(_QWORD *, _QWORD *, __int64))v9)(v8, v8, 3);
  return result;
}
