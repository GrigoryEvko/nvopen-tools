// Function: sub_11FAA70
// Address: 0x11faa70
//
__int64 __fastcall sub_11FAA70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v12[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( !qword_4F92310
    || (v6 = sub_BD5D20(a3),
        v12[1] = v7,
        v12[0] = (__int64)v6,
        sub_C931B0(v12, (_WORD *)qword_4F92308, qword_4F92310, 0) != -1) )
  {
    v8 = sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8;
    v9 = sub_BC1CD0(a4, &unk_4F8E5A8, a3);
    v10 = sub_11FCC10(a3, v8);
    sub_11FA890(a3, v8, v9 + 8, v10, 1);
  }
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
