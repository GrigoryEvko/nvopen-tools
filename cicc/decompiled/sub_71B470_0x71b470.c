// Function: sub_71B470
// Address: 0x71b470
//
__int64 __fastcall sub_71B470(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rax
  __int64 v9; // r13
  __int64 v10; // rax
  _QWORD *v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v16[7]; // [rsp+8h] [rbp-38h] BYREF

  v6 = 0;
  v8 = sub_69C8B0(a1, a2, a3, v16, a5, a6);
  if ( *(_BYTE *)(v8 + 177) == 2 )
  {
    v9 = v8;
    v10 = sub_726B30(17);
    *(_QWORD *)(v10 + 24) = a4;
    v6 = v10;
    *(_QWORD *)(v10 + 72) = *(_QWORD *)(v9 + 184);
    v11 = (_QWORD *)sub_726B30(8);
    *v11 = *(_QWORD *)dword_4F07508;
    v11[6] = sub_73E830(v9);
    v12 = (_QWORD *)sub_726B30(1);
    v13 = *(_QWORD *)dword_4F07508;
    v12[3] = a4;
    *v12 = v13;
    v14 = v16[0];
    v12[9] = v11;
    v12[6] = v14;
    v11[3] = v12;
    *(_QWORD *)(v6 + 16) = v12;
  }
  return v6;
}
