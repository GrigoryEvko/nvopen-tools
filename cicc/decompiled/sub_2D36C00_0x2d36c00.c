// Function: sub_2D36C00
// Address: 0x2d36c00
//
void __fastcall sub_2D36C00(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // r14
  _QWORD *v7; // r12
  __int64 v8; // rax
  unsigned __int8 *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r12
  _QWORD *v12; // rax
  unsigned int v13[4]; // [rsp+10h] [rbp-60h] BYREF
  char v14; // [rsp+20h] [rbp-50h]
  __int64 v15[8]; // [rsp+30h] [rbp-40h] BYREF

  v15[0] = a1;
  v15[1] = a3;
  v15[2] = a4;
  v15[3] = sub_B10CD0(a3 + 48);
  if ( a2 )
  {
    if ( a2 != 1 )
    {
      sub_2D36AC0(v15, 0, *(_QWORD *)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 24LL));
      return;
    }
    goto LABEL_5;
  }
  v5 = sub_2D28480(a3);
  if ( (unsigned __int8)sub_B59AF0(v5) )
  {
LABEL_5:
    sub_2D36AC0(
      v15,
      *(_QWORD **)(*(_QWORD *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) + 24LL),
      *(_QWORD *)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 24LL));
    return;
  }
  v6 = sub_B595C0(v5);
  v7 = *(_QWORD **)(*(_QWORD *)(v5 + 32 * (5LL - (*(_DWORD *)(v5 + 4) & 0x7FFFFFF))) + 24LL);
  v8 = *(_QWORD *)(*(_QWORD *)(a3 + 32 * (2LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))) + 24LL);
  sub_AF47B0((__int64)v13, *(unsigned __int64 **)(v8 + 16), *(unsigned __int64 **)(v8 + 24));
  if ( v14 )
    v7 = (_QWORD *)sub_B0E470((__int64)v7, v13[2], v13[0]);
  v9 = sub_2D27AA0(*(_QWORD *)(a1 + 128), v6, v7);
  v11 = v10;
  v12 = sub_B98A20((__int64)v9, v6);
  sub_2D36AC0(v15, v12, v11);
}
