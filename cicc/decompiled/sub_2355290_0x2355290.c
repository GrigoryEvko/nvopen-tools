// Function: sub_2355290
// Address: 0x2355290
//
__int64 __fastcall sub_2355290(__int64 a1, char a2, char a3, char a4)
{
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  unsigned __int64 v10[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
    *v6 = &unk_4A122B8;
  *(_BYTE *)(a1 + 50) = a4;
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = a2;
  *(_BYTE *)(a1 + 49) = a3;
  *(_BYTE *)(a1 + 51) = 0;
  v7 = (_QWORD *)sub_22077B0(0x10u);
  if ( v7 )
    *v7 = &unk_4A0B640;
  v10[0] = (unsigned __int64)v7;
  sub_2353900((unsigned __int64 *)(a1 + 8), v10);
  if ( v10[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  v8 = (_QWORD *)sub_22077B0(0x10u);
  if ( v8 )
    *v8 = &unk_4A0B680;
  v10[0] = (unsigned __int64)v8;
  sub_2353900((unsigned __int64 *)(a1 + 8), v10);
  if ( v10[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v10[0] + 8LL))(v10[0]);
  return a1;
}
