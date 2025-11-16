// Function: sub_137ED00
// Address: 0x137ed00
//
__int64 __fastcall sub_137ED00(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v14; // [rsp+10h] [rbp-50h]
  char v15; // [rsp+20h] [rbp-40h] BYREF

  v4 = sub_157EBA0(a2);
  if ( (unsigned int)sub_15F4D60(v4) == 1 )
    goto LABEL_5;
  if ( !*(_QWORD *)(v4 + 48) && *(__int16 *)(v4 + 18) >= 0 )
    goto LABEL_8;
  v5 = sub_1625790(v4, 2);
  v6 = v5;
  if ( !v5 )
    goto LABEL_8;
  v7 = sub_161E970(*(_QWORD *)(v5 - 8LL * *(unsigned int *)(v5 + 8)));
  if ( v8 != 14
    || *(_QWORD *)v7 != 0x775F68636E617262LL
    || *(_DWORD *)(v7 + 8) != 1751607653
    || *(_WORD *)(v7 + 12) != 29556 )
  {
LABEL_5:
    *(_QWORD *)a1 = a1 + 16;
    sub_137E9E0((__int64 *)a1, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v10 = *(unsigned int *)(v6 + 8);
  v11 = (unsigned int)(a3 + 1);
  if ( (unsigned int)v11 < (unsigned int)v10 )
  {
    v12 = *(_QWORD *)(v6 + 8 * (v11 - v10));
    if ( *(_BYTE *)v12 == 1 && *(_BYTE *)(*(_QWORD *)(v12 + 136) + 16LL) == 13 )
    {
      v13[0] = &v15;
      v13[1] = "\"";
      v14 = 770;
      sub_16E2FC0(a1, v13);
      return a1;
    }
  }
LABEL_8:
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  return a1;
}
