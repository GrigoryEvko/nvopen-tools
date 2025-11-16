// Function: sub_39A2FF0
// Address: 0x39a2ff0
//
__int64 __fastcall sub_39A2FF0(_QWORD *a1, char a2)
{
  __int64 v3; // rdi
  void (*v4)(); // rax
  __int64 v5; // rdi
  void (*v6)(); // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  _QWORD v10[2]; // [rsp+0h] [rbp-30h] BYREF
  char v11; // [rsp+10h] [rbp-20h]
  char v12; // [rsp+11h] [rbp-1Fh]

  sub_39A2D30((__int64)a1, a2, *(_BYTE *)(a1[25] + 4513LL) == 0 ? 2 : 6);
  v3 = *(_QWORD *)(a1[24] + 256LL);
  v4 = *(void (**)())(*(_QWORD *)v3 + 104LL);
  v12 = 1;
  v10[0] = "Type Signature";
  v11 = 3;
  if ( v4 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v4)(v3, v10, 1);
    v3 = *(_QWORD *)(a1[24] + 256LL);
  }
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v3 + 424LL))(v3, a1[75], 8);
  v5 = *(_QWORD *)(a1[24] + 256LL);
  v6 = *(void (**)())(*(_QWORD *)v5 + 104LL);
  v12 = 1;
  v10[0] = "Type DIE Offset";
  v11 = 3;
  if ( v6 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD *, __int64))v6)(v5, v10, 1);
    v5 = *(_QWORD *)(a1[24] + 256LL);
  }
  v7 = 0;
  v8 = a1[76];
  if ( v8 )
    v7 = *(unsigned int *)(v8 + 16);
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v5 + 424LL))(v5, v7, 4);
}
