// Function: sub_38D01D0
// Address: 0x38d01d0
//
__int64 __fastcall sub_38D01D0(__int64 a1, __int64 a2, char a3, _QWORD *a4)
{
  unsigned __int64 v6; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rsi
  _QWORD *v12; // rax
  unsigned __int64 *v13; // rax
  _QWORD v14[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-70h] BYREF
  __int16 v16; // [rsp+20h] [rbp-60h]
  _QWORD v17[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 )
  {
    v11 = *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_9:
    *a4 = *(_QWORD *)(a2 + 24) + sub_38D01B0(a1, v11);
    return 1;
  }
  if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
  {
    *(_BYTE *)(a2 + 8) |= 4u;
    v9 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a2 + 24));
    v10 = v9 | *(_QWORD *)a2 & 7LL;
    *(_QWORD *)a2 = v10;
    if ( v9 )
    {
      v11 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v11 )
      {
        v11 = 0;
        if ( (*(_BYTE *)(a2 + 9) & 0xC) == 8 )
        {
          *(_BYTE *)(a2 + 8) |= 4u;
          v11 = (__int64)sub_38CE440(*(_QWORD *)(a2 + 24));
          *(_QWORD *)a2 = v11 | *(_QWORD *)a2 & 7LL;
        }
      }
      goto LABEL_9;
    }
  }
  if ( a3 )
  {
    v12 = 0;
    if ( (*(_BYTE *)a2 & 4) != 0 )
    {
      v13 = *(unsigned __int64 **)(a2 - 8);
      v6 = *v13;
      v12 = v13 + 2;
    }
    v14[0] = v12;
    v17[0] = "unable to evaluate offset to undefined symbol '";
    v17[1] = v14;
    v15[0] = v17;
    v14[1] = v6;
    v18 = 1283;
    v15[1] = "'";
    v16 = 770;
    sub_16BCFB0((__int64)v15, 1u);
  }
  return 0;
}
