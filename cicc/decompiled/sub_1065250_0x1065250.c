// Function: sub_1065250
// Address: 0x1065250
//
__int64 __fastcall sub_1065250(__int64 **a1, unsigned __int8 *a2)
{
  unsigned __int8 *v2; // rax
  unsigned __int8 *v3; // rax
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v8; // r13
  const char *v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r15
  _QWORD *v17; // rax

  v2 = (unsigned __int8 *)sub_AD69F0(a2, 2);
  v3 = sub_BD3990(v2, 2);
  if ( *v3 <= 3u )
  {
    v4 = *a1;
    v5 = (__int64)v3;
    if ( (v3[7] & 0x10) == 0 )
      goto LABEL_12;
    v6 = 0;
    if ( (v3[32] & 0xFu) - 7 > 1 )
    {
      v8 = *v4;
      v9 = sub_BD5D20((__int64)v3);
      v11 = sub_BA8B30(v8, (__int64)v9, v10);
      v6 = v11;
      if ( !v11 || (*(_BYTE *)(v11 + 32) & 0xFu) - 7 <= 1 )
      {
        v4 = *a1;
        v6 = 0;
        return (unsigned int)sub_1061370((__int64)v4, v6, v5) ^ 1;
      }
      if ( *(_BYTE *)v11 || (*(_BYTE *)(v11 + 33) & 0x20) == 0 || *(_BYTE *)v5 )
      {
        v4 = *a1;
        return (unsigned int)sub_1061370((__int64)v4, v6, v5) ^ 1;
      }
      v16 = *(_QWORD **)(v11 + 24);
      v17 = sub_10651F0((__int64)(v4 + 6), *(_QWORD *)(v5 + 24), v12, v13, v14, v15);
      v4 = *a1;
      if ( v16 != v17 )
LABEL_12:
        v6 = 0;
    }
    return (unsigned int)sub_1061370((__int64)v4, v6, v5) ^ 1;
  }
  return 0;
}
