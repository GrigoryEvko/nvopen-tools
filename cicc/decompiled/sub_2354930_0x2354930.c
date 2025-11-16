// Function: sub_2354930
// Address: 0x2354930
//
unsigned __int64 __fastcall sub_2354930(__int64 a1, _QWORD *a2, char a3, char a4, char a5, char a6)
{
  unsigned __int64 *v6; // r12
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v10[3]; // [rsp+8h] [rbp-18h] BYREF

  v6 = (unsigned __int64 *)(a1 + 8);
  *(_QWORD *)a1 = *a2;
  *a2 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = a3;
  *(_BYTE *)(a1 + 49) = a4;
  *(_BYTE *)(a1 + 50) = a5;
  *(_BYTE *)(a1 + 51) = a6;
  v7 = (_QWORD *)sub_22077B0(0x10u);
  if ( v7 )
    *v7 = &unk_4A0B640;
  v10[0] = (unsigned __int64)v7;
  sub_2353900(v6, v10);
  if ( v10[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0B680;
  v10[0] = (unsigned __int64)v8;
  result = sub_2353900(v6, v10);
  if ( v10[0] )
    return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  return result;
}
