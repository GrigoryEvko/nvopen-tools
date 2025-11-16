// Function: sub_2C13EA0
// Address: 0x2c13ea0
//
__int64 __fastcall sub_2C13EA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // ebx
  _QWORD *v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // [rsp+8h] [rbp-68h]
  _QWORD v16[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v17; // [rsp+30h] [rbp-40h]

  BYTE4(v15) = *(_BYTE *)(a3 + 8) == 18;
  LODWORD(v15) = *(_DWORD *)(a3 + 32);
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 8) + 24LL);
  if ( (unsigned __int8)sub_B50C50(v6, *(_QWORD *)(a3 + 24), a4) )
  {
    v17 = 257;
    return sub_10E0940(a1, a2, a3, (__int64)v16);
  }
  else
  {
    v8 = sub_9208B0(a4, v6);
    v16[1] = v9;
    v16[0] = v8;
    v10 = sub_CA1930(v16);
    v11 = (_QWORD *)sub_BD5C60(a2);
    v12 = (__int64 *)sub_BCD140(v11, v10);
    v13 = sub_BCE1B0(v12, v15);
    v17 = 257;
    v14 = sub_10E0940(a1, a2, v13, (__int64)v16);
    v17 = 257;
    return sub_10E0940(a1, v14, a3, (__int64)v16);
  }
}
