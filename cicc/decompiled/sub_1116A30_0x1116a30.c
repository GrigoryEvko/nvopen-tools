// Function: sub_1116A30
// Address: 0x1116a30
//
_QWORD *__fastcall sub_1116A30(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  __int64 v4; // rax
  int v7; // edx
  int v8; // r14d
  __int64 v9; // r15
  unsigned int v10; // ebx
  __int64 v11; // rbx
  _QWORD **v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // [rsp+18h] [rbp-68h]
  _BYTE v17[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v3 = 0;
  v4 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v4 != 85 )
    return v3;
  v3 = *(_QWORD **)(v4 - 32);
  if ( !v3 )
    return v3;
  if ( *(_BYTE *)v3 || v3[3] != *(_QWORD *)(v4 + 80) || (*((_BYTE *)v3 + 33) & 0x20) == 0 )
    return 0;
  v7 = *((_DWORD *)v3 + 9);
  v3 = 0;
  if ( (unsigned int)(v7 - 180) > 1 )
    return v3;
  v8 = *(_WORD *)(a2 + 2) & 0x3F;
  if ( (unsigned int)(v8 - 32) > 1 )
    return v3;
  v9 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
  if ( *(_QWORD *)(v4 + 32 * (1LL - (*(_DWORD *)(v4 + 4) & 0x7FFFFFF))) != v9 )
    return v3;
  v10 = *(_DWORD *)(a3 + 8);
  if ( v10 > 0x40 )
  {
    if ( v10 != (unsigned int)sub_C444A0(a3) )
    {
      v3 = 0;
      if ( v10 != (unsigned int)sub_C445E0(a3) )
        return v3;
    }
    goto LABEL_14;
  }
  if ( *(_QWORD *)a3 && v10 && *(_QWORD *)a3 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) )
    return 0;
LABEL_14:
  v11 = *(_QWORD *)(a2 - 32);
  v18 = 257;
  v3 = sub_BD2C40(72, unk_3F10FD0);
  if ( v3 )
  {
    v12 = *(_QWORD ***)(v9 + 8);
    v13 = *((unsigned __int8 *)v12 + 8);
    if ( (unsigned int)(v13 - 17) > 1 )
    {
      v15 = sub_BCB2A0(*v12);
    }
    else
    {
      BYTE4(v16) = (_BYTE)v13 == 18;
      LODWORD(v16) = *((_DWORD *)v12 + 8);
      v14 = (__int64 *)sub_BCB2A0(*v12);
      v15 = sub_BCE1B0(v14, v16);
    }
    sub_B523C0((__int64)v3, v15, 53, v8, v9, v11, (__int64)v17, 0, 0, 0);
  }
  return v3;
}
