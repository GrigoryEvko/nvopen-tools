// Function: sub_120BD00
// Address: 0x120bd00
//
__int64 __fastcall sub_120BD00(__int64 a1, _DWORD *a2)
{
  unsigned int v3; // r12d
  unsigned __int64 v4; // rsi
  unsigned int v5; // r12d
  unsigned int v7; // r13d
  __int64 v8; // rdi
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  _QWORD v11[4]; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+30h] [rbp-30h]
  char v13; // [rsp+31h] [rbp-2Fh]

  if ( *(_DWORD *)(a1 + 240) != 529 || (v3 = *(unsigned __int8 *)(a1 + 332), !(_BYTE)v3) )
  {
    v4 = *(_QWORD *)(a1 + 232);
    v13 = 1;
    v11[0] = "expected integer";
    v5 = 1;
    v12 = 3;
    sub_11FD800(a1 + 176, v4, (__int64)v11, 1);
    return v5;
  }
  v7 = *(_DWORD *)(a1 + 328);
  if ( v7 <= 0x40 )
  {
    v10 = *(_QWORD *)(a1 + 320);
    v8 = a1 + 176;
    if ( v10 > 0x100000000LL )
      goto LABEL_8;
  }
  else if ( v7 - (unsigned int)sub_C444A0(a1 + 320) > 0x40 || (v10 = **(_QWORD **)(a1 + 320), v10 > 0x100000000LL) )
  {
    v8 = a1 + 176;
    goto LABEL_8;
  }
  v8 = a1 + 176;
  if ( (unsigned int)v10 == v10 )
  {
    *a2 = v10;
    v5 = 0;
    *(_DWORD *)(a1 + 240) = sub_1205200(v8);
    return v5;
  }
LABEL_8:
  v9 = *(_QWORD *)(a1 + 232);
  v11[0] = "expected 32-bit integer (too large)";
  v13 = 1;
  v12 = 3;
  sub_11FD800(v8, v9, (__int64)v11, 1);
  return v3;
}
