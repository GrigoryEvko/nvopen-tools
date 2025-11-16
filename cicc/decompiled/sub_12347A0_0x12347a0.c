// Function: sub_12347A0
// Address: 0x12347a0
//
__int64 __fastcall sub_12347A0(__int64 a1, __int64 *a2, __int64 *a3, int a4, char a5)
{
  unsigned __int64 v8; // r14
  unsigned int v9; // r10d
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rax
  int v14; // ecx
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v17[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v15, a3)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in arithmetic operation")
    || (unsigned __int8)sub_1224B80((__int64 **)a1, *(_QWORD *)(v15 + 8), &v16, a3) )
  {
    return 1;
  }
  v11 = *(_QWORD *)(v15 + 8);
  if ( !a5 )
  {
    v14 = *(unsigned __int8 *)(v11 + 8);
    if ( (unsigned int)(v14 - 17) <= 1 )
      LOBYTE(v14) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
    if ( (_BYTE)v14 != 12 )
      goto LABEL_17;
LABEL_10:
    v18 = 257;
    v13 = sub_B504D0(a4, v15, v16, (__int64)v17, 0, 0);
    v9 = 0;
    *a2 = v13;
    return v9;
  }
  v12 = *(unsigned __int8 *)(v11 + 8);
  if ( (unsigned int)(v12 - 17) <= 1 )
    LOBYTE(v12) = *(_BYTE *)(**(_QWORD **)(v11 + 16) + 8LL);
  if ( (unsigned __int8)v12 <= 3u || (_BYTE)v12 == 5 || (v12 & 0xFD) == 4 )
    goto LABEL_10;
LABEL_17:
  v18 = 259;
  v17[0] = "invalid operand type for instruction";
  sub_11FD800(a1 + 176, v8, (__int64)v17, 1);
  return 1;
}
