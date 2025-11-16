// Function: sub_2355FA0
// Address: 0x2355fa0
//
__int64 __fastcall sub_2355FA0(__int64 a1, __int64 *a2, char a3, char a4, char a5)
{
  __int64 v8; // rbx
  _QWORD *v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v8 = *a2;
  v9 = (_QWORD *)sub_22077B0(0x10u);
  if ( v9 )
  {
    v9[1] = v8;
    *v9 = &unk_4A120B8;
  }
  *(_BYTE *)(a1 + 50) = a5;
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = a3;
  *(_BYTE *)(a1 + 49) = a4;
  *(_BYTE *)(a1 + 51) = 0;
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0B640;
  v13[0] = (unsigned __int64)v10;
  sub_2353900((unsigned __int64 *)(a1 + 8), v13);
  if ( v13[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v13[0] + 8LL))(v13[0]);
  v11 = (_QWORD *)sub_22077B0(0x10u);
  if ( v11 )
    *v11 = &unk_4A0B680;
  v13[0] = (unsigned __int64)v11;
  sub_2353900((unsigned __int64 *)(a1 + 8), v13);
  if ( v13[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v13[0] + 8LL))(v13[0]);
  return a1;
}
