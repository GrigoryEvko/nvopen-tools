// Function: sub_2356430
// Address: 0x2356430
//
__int64 __fastcall sub_2356430(__int64 a1, __int64 *a2, char a3, char a4, char a5)
{
  __int64 v7; // r13
  __int16 v8; // bx
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v7 = *a2;
  v8 = *((_WORD *)a2 + 4);
  v9 = sub_22077B0(0x18u);
  if ( v9 )
  {
    *(_QWORD *)(v9 + 8) = v7;
    *(_WORD *)(v9 + 16) = v8;
    *(_QWORD *)v9 = &unk_4A12478;
  }
  *(_QWORD *)a1 = v9;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = a3;
  *(_BYTE *)(a1 + 49) = a4;
  *(_BYTE *)(a1 + 50) = a5;
  *(_BYTE *)(a1 + 51) = 0;
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0B640;
  v14[0] = (unsigned __int64)v10;
  sub_2353900((unsigned __int64 *)(a1 + 8), v14);
  if ( v14[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
  v11 = (_QWORD *)sub_22077B0(0x10u);
  if ( v11 )
    *v11 = &unk_4A0B680;
  v14[0] = (unsigned __int64)v11;
  sub_2353900((unsigned __int64 *)(a1 + 8), v14);
  if ( v14[0] )
    (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
  return a1;
}
