// Function: sub_6F8AB0
// Address: 0x6f8ab0
//
__int64 __fastcall sub_6F8AB0(__int64 *a1, _QWORD *a2, _QWORD *a3, _QWORD *a4, _QWORD *a5, _DWORD *a6, _QWORD *a7)
{
  __int64 v8; // rbx
  char v9; // r11
  char v10; // r15
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  bool v15; // r11
  __int64 v16; // r13
  __int64 v17; // r15
  __int64 *v18; // r15
  __int64 *v19; // rdi
  __int64 result; // rax
  __int64 v21; // r15
  bool v24; // [rsp+1Fh] [rbp-1C1h]
  _BYTE v27[432]; // [rsp+30h] [rbp-1B0h] BYREF

  v8 = *a1;
  v9 = *(_BYTE *)(qword_4D03C50 + 20LL);
  v24 = (v9 & 8) != 0;
  *(_BYTE *)(qword_4D03C50 + 20LL) = v9 & 0xF7;
  v10 = *(_BYTE *)(v8 + 24);
  v11 = sub_6E3DA0(v8, (__int64)v27);
  v15 = v24;
  v16 = v11;
  if ( v10 == 1 )
  {
    v21 = *(_QWORD *)(v8 + 72);
    if ( a2 )
    {
      sub_6F85E0(*(__int64 **)(v8 + 72), (__int64)a1, 4 * (unsigned int)(*(_BYTE *)(v8 + 56) == 0), a2, 0, v14);
      v18 = *(__int64 **)(v21 + 16);
      v15 = v24;
      if ( !v18 )
        goto LABEL_10;
    }
    else
    {
      v18 = *(__int64 **)(v21 + 16);
      if ( !v18 )
        goto LABEL_10;
    }
    if ( *(_BYTE *)(v8 + 56) == 91 && v15 )
      *(_BYTE *)(qword_4D03C50 + 20LL) |= 8u;
    goto LABEL_5;
  }
  v17 = *(_QWORD *)(v8 + 64);
  if ( a2 )
    sub_6F85E0(*(__int64 **)(v8 + 64), (__int64)a1, 0, a2, 0, v14);
  v18 = *(__int64 **)(v17 + 16);
  if ( v18 )
  {
LABEL_5:
    if ( a3 )
      sub_6F8800(v18, (__int64)a1, a3, v12, v13, v14);
    *(_BYTE *)(qword_4D03C50 + 20LL) &= ~8u;
    v19 = (__int64 *)v18[2];
    if ( v19 && a4 )
      sub_6F8800(v19, (__int64)a1, a4, v12, v13, v14);
  }
LABEL_10:
  *a5 = *(_QWORD *)(v16 + 356);
  *a6 = *(_DWORD *)(v16 + 364);
  if ( a7 )
    *a7 = *(_QWORD *)(v16 + 368);
  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x20) != 0 && v8 == unk_4D03C40 )
    *(_BYTE *)(qword_4D03C50 + 20LL) |= 8u;
  return result;
}
