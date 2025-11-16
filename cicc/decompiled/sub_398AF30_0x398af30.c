// Function: sub_398AF30
// Address: 0x398af30
//
void __fastcall sub_398AF30(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // r8
  void (*v6)(); // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  void (__fastcall **v9[2])(_QWORD, _QWORD, _QWORD); // [rsp+0h] [rbp-40h] BYREF
  char v10; // [rsp+10h] [rbp-30h]
  char v11; // [rsp+11h] [rbp-2Fh]

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(v4 + 256);
  v6 = *(void (**)())(*(_QWORD *)v5 + 104LL);
  v11 = 1;
  v9[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD))"Loc expr size";
  v10 = 3;
  if ( v6 != nullsub_580 )
  {
    ((void (__fastcall *)(__int64, _QWORD, __int64))v6)(v5, v9, 1);
    v4 = *(_QWORD *)(a1 + 8);
  }
  v7 = *(_QWORD *)(a2 + 16);
  if ( ((a2 - *(_QWORD *)(a1 + 1336)) >> 5) + 1 == *(_DWORD *)(a1 + 1344) )
    LODWORD(v8) = *(_DWORD *)(a1 + 2384);
  else
    v8 = *(_QWORD *)(a2 + 48);
  sub_396F320(v4, v8 - v7);
  v9[1] = *(void (__fastcall ***)(_QWORD, _QWORD, _QWORD))(a1 + 8);
  v9[0] = (void (__fastcall **)(_QWORD, _QWORD, _QWORD))&unk_4A3F910;
  sub_398A890(a1, v9, a2);
}
