// Function: sub_11E97B0
// Address: 0x11e97b0
//
__int64 __fastcall sub_11E97B0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 *v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 result; // rax
  __int64 *v13; // [rsp+8h] [rbp-38h]

  sub_11E6850((__int64)a1, (unsigned __int8 *)a2, a3, 1);
  if ( (unsigned __int8)sub_11F3070(*(_QWORD *)(a2 + 40), a1[9], a1[8], 0) )
    return 0;
  if ( *(_QWORD *)(a2 + 16) )
    return 0;
  v4 = sub_98B430(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 8u);
  if ( !v4 )
    return 0;
  v5 = (__int64 *)a1[3];
  v6 = sub_B43CA0(a2);
  LODWORD(v5) = sub_97FA80(*v5, v6);
  v7 = (_QWORD *)sub_BD5C60(a2);
  v8 = sub_BCCE00(v7, (unsigned int)v5);
  v9 = a1[2];
  v13 = (__int64 *)a1[3];
  v10 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  v11 = sub_AD64C0(v8, v4 - 1, 0);
  result = sub_11CB6D0(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), v11, v10, a3, v9, v13);
  if ( !result )
    return 0;
  if ( *(_BYTE *)result == 85 )
    *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
  return result;
}
