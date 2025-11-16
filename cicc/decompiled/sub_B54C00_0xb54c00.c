// Function: sub_B54C00
// Address: 0xb54c00
//
__int64 __fastcall sub_B54C00(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  __int16 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rdx
  int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v10; // [rsp+8h] [rbp-58h]
  _BYTE v11[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v12; // [rsp+30h] [rbp-30h]

  v1 = *(_QWORD *)(a1 - 64);
  v2 = *(_QWORD *)(a1 - 32);
  v12 = 257;
  v3 = *(_WORD *)(a1 + 2) & 0x3F;
  v4 = sub_BD2C40(72, unk_3F10FD0);
  if ( v4 )
  {
    v5 = *(_QWORD **)(v1 + 8);
    v6 = *((unsigned __int8 *)v5 + 8);
    if ( (unsigned int)(v6 - 17) > 1 )
    {
      v8 = sub_BCB2A0(*v5);
    }
    else
    {
      BYTE4(v10) = (_BYTE)v6 == 18;
      LODWORD(v10) = *((_DWORD *)v5 + 8);
      v7 = sub_BCB2A0(*v5);
      v8 = sub_BCE1B0(v7, v10);
    }
    sub_B523C0(v4, v8, 53, v3, v1, v2, (__int64)v11, 0, 0, 0);
  }
  return v4;
}
