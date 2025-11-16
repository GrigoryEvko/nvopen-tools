// Function: sub_3900970
// Address: 0x3900970
//
__int64 __fastcall sub_3900970(__int64 a1, __int64 a2)
{
  _DWORD *v2; // r12
  __int64 v4; // rax
  __int64 v5; // r8
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // rcx
  bool v10; // di
  __int64 v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  unsigned __int64 v16; // [rsp+8h] [rbp-58h]
  _QWORD v17[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v18; // [rsp+20h] [rbp-40h]
  _QWORD v19[2]; // [rsp+30h] [rbp-30h] BYREF
  __int16 v20; // [rsp+40h] [rbp-20h]

  v2 = (_DWORD *)a2;
  v4 = sub_3909460(*(_QWORD *)(a1 + 8));
  v6 = v4;
  if ( *(_DWORD *)v4 == 2 )
  {
    v9 = *(_QWORD *)(v4 + 8);
    v7 = *(_QWORD *)(v4 + 16);
  }
  else
  {
    v7 = *(_QWORD *)(v4 + 16);
    if ( !v7 )
    {
      v9 = *(_QWORD *)(v6 + 8);
      v16 = 0;
      v15 = v9;
LABEL_17:
      v10 = 0;
      v12 = 1;
      v5 = 0;
      v11 = 1;
      goto LABEL_10;
    }
    a2 = 1;
    v8 = v7 - 1;
    if ( v7 == 1 )
      v8 = 1;
    if ( v8 <= v7 )
      v7 = v8;
    --v7;
    v9 = *(_QWORD *)(v6 + 8) + 1LL;
  }
  v15 = v9;
  v16 = v7;
  v10 = v7 == 7;
  if ( v7 != 8 )
  {
    switch ( v7 )
    {
      case 7uLL:
        if ( *(_DWORD *)v9 == 1668508004 && *(_WORD *)(v9 + 4) == 29281 && *(_BYTE *)(v9 + 6) == 100 )
        {
          v10 = 1;
          v12 = 0;
          v5 = 1;
          v11 = 2;
        }
        else
        {
          v10 = 1;
          v12 = 1;
          v5 = 0;
          v11 = 2;
        }
        goto LABEL_10;
      case 9uLL:
        if ( *(_QWORD *)v9 == 0x7A69735F656D6173LL && *(_BYTE *)(v9 + 8) == 101 )
        {
          v10 = 0;
          v11 = 3;
          v12 = 0;
          v5 = 1;
LABEL_12:
          if ( (v10 & (unsigned __int8)v12) == 0 )
            goto LABEL_13;
          if ( *(_DWORD *)v9 == 1735549292 && *(_WORD *)(v9 + 4) == 29541 && *(_BYTE *)(v9 + 6) == 116 )
          {
            v11 = 6;
            goto LABEL_31;
          }
LABEL_14:
          if ( !(_BYTE)v5 )
          {
LABEL_15:
            v18 = 1283;
            *v2 = 0;
            v13 = *(_QWORD *)(a1 + 8);
            v17[0] = "unrecognized COMDAT type '";
            v17[1] = &v15;
            v19[0] = v17;
            v20 = 770;
            v19[1] = "'";
            return sub_3909CF0(v13, v19, 0, 0, v5, v11);
          }
          goto LABEL_31;
        }
        a2 = 0;
        v10 = 0;
        v12 = 1;
        v5 = 0;
        v11 = 1;
        goto LABEL_11;
      case 0xDuLL:
        if ( *(_QWORD *)v9 != 0x6E6F635F656D6173LL || *(_DWORD *)(v9 + 8) != 1953391988 || *(_BYTE *)(v9 + 12) != 115 )
        {
          v5 = 0;
          v11 = 1;
          v12 = 0;
          goto LABEL_14;
        }
        v11 = 4;
        v5 = 1;
        LODWORD(v12) = 0;
        goto LABEL_13;
    }
    goto LABEL_17;
  }
  v12 = 0x796C6E6F5F656E6FLL;
  v11 = 1;
  LOBYTE(v12) = *(_QWORD *)v9 != 0x796C6E6F5F656E6FLL;
  LOBYTE(v5) = *(_QWORD *)v9 == 0x796C6E6F5F656E6FLL;
LABEL_10:
  LOBYTE(a2) = v7 == 11;
  a2 = (unsigned int)v12 & (unsigned int)a2;
LABEL_11:
  if ( !(_BYTE)a2 )
    goto LABEL_12;
  a2 = 0x746169636F737361LL;
  if ( *(_QWORD *)v9 == 0x746169636F737361LL && *(_WORD *)(v9 + 8) == 30313 && *(_BYTE *)(v9 + 10) == 101 )
  {
    v11 = 5;
LABEL_31:
    *v2 = v11;
    goto LABEL_32;
  }
LABEL_13:
  LOBYTE(v7) = v7 == 6;
  v12 = (unsigned int)v7 & (unsigned int)v12;
  if ( !(_BYTE)v12 )
    goto LABEL_14;
  if ( *(_DWORD *)v9 != 1702323566 || *(_WORD *)(v9 + 4) != 29811 )
    goto LABEL_15;
  *v2 = 7;
LABEL_32:
  (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64, __int64, __int64, __int64, unsigned __int64))(**(_QWORD **)(a1 + 8) + 136LL))(
    *(_QWORD *)(a1 + 8),
    a2,
    v12,
    v9,
    v5,
    v11,
    v15,
    v16);
  return 0;
}
