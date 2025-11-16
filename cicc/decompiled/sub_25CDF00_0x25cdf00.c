// Function: sub_25CDF00
// Address: 0x25cdf00
//
bool __fastcall sub_25CDF00(__int64 a1)
{
  char v1; // al
  __int64 v2; // rdx
  __int64 v3; // rcx
  char v4; // cl
  bool result; // al
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r15
  bool v9; // zf
  __int16 v10; // bx
  unsigned __int8 *v11; // r13
  unsigned int v12; // ebx
  __int64 v13; // rax
  unsigned int v14; // ebx
  __int64 v15; // [rsp+8h] [rbp-68h]
  _BYTE v16[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v17; // [rsp+30h] [rbp-40h]

  if ( !*(_BYTE *)a1 )
  {
    sub_B2CA40(a1, 0);
LABEL_3:
    v1 = *(_BYTE *)(a1 + 32);
    *(_BYTE *)(a1 + 32) = v1 & 0xF0;
    if ( (v1 & 0x30) != 0 )
      *(_BYTE *)(a1 + 33) |= 0x40u;
    sub_B91E30(a1, 0);
    sub_B2F990(a1, 0, v2, v3);
    v4 = *(_BYTE *)(a1 + 32);
    result = 1;
    if ( (v4 & 0xFu) - 7 > 1 )
    {
      result = (v4 & 0xF) != 9 && (v4 & 0x30) != 0;
      if ( !result )
      {
        *(_BYTE *)(a1 + 33) &= ~0x40u;
        return 1;
      }
    }
    return result;
  }
  if ( *(_BYTE *)a1 == 3 )
  {
    sub_B30160(a1, 0);
    goto LABEL_3;
  }
  v6 = *(_QWORD *)(a1 + 24);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_QWORD *)(a1 + 40);
  v9 = *(_BYTE *)(v6 + 8) == 13;
  v17 = 257;
  if ( v9 )
  {
    v12 = *(_DWORD *)(v7 + 8);
    v13 = sub_BD2DA0(136);
    v14 = v12 >> 8;
    v11 = (unsigned __int8 *)v13;
    if ( v13 )
      sub_B2C3B0(v13, v6, 0, v14, (__int64)v16, v8);
  }
  else
  {
    BYTE4(v15) = 1;
    LODWORD(v15) = *(_DWORD *)(v7 + 8) >> 8;
    v10 = (*(_BYTE *)(a1 + 33) >> 2) & 7;
    v11 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
    if ( v11 )
      sub_B30000((__int64)v11, v8, (_QWORD *)v6, 0, 0, 0, (__int64)v16, 0, v10, v15, 0);
  }
  sub_BD6B90(v11, (unsigned __int8 *)a1);
  sub_BD84D0(a1, (__int64)v11);
  return 0;
}
