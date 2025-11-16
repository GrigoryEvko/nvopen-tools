// Function: sub_2FE6750
// Address: 0x2fe6750
//
__int64 __fastcall sub_2FE6750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v5)(__int64, __int64); // rax
  int v6; // eax
  unsigned __int16 v7; // bx
  __int64 v9; // rax
  char v10; // dl
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-60h] BYREF
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  char v19; // [rsp+18h] [rbp-48h]
  __int64 v20; // [rsp+20h] [rbp-40h] BYREF
  char v21; // [rsp+28h] [rbp-38h]
  __int64 v22; // [rsp+30h] [rbp-30h]
  __int64 v23; // [rsp+38h] [rbp-28h]

  v16 = a2;
  v17 = a3;
  if ( (_WORD)a2 )
  {
    if ( (unsigned __int16)(a2 - 17) > 0xD3u )
      goto LABEL_3;
    return v16;
  }
  if ( (unsigned __int8)sub_30070B0(&v16, a2, a3) )
    return v16;
LABEL_3:
  v5 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 56LL);
  if ( v5 == sub_2FE48D0 )
  {
    v6 = sub_AE2980(a4, 0)[1];
    if ( v6 == 1 )
    {
      v7 = 2;
      v6 = 2;
    }
    else if ( v6 == 2 )
    {
      v7 = 3;
      v6 = 3;
    }
    else
    {
      v7 = 4;
      if ( v6 != 4 )
      {
        switch ( v6 )
        {
          case 8:
            v7 = 5;
            v6 = 5;
            break;
          case 16:
            v7 = 6;
            v6 = 6;
            break;
          case 32:
            v7 = 7;
            v6 = 7;
            break;
          case 64:
            v7 = 8;
            v6 = 8;
            break;
          case 128:
            v7 = 9;
            v6 = 9;
            break;
          default:
            goto LABEL_33;
        }
      }
    }
  }
  else
  {
    v6 = ((unsigned __int16 (__fastcall *)(__int64, __int64, _QWORD, __int64))v5)(a1, a4, (unsigned int)v16, v17);
    v7 = v6;
    if ( (unsigned __int16)v6 <= 1u || (unsigned __int16)(v6 - 504) <= 7u )
      goto LABEL_33;
  }
  v9 = 16LL * (v6 - 1);
  v10 = byte_444C4A0[v9 + 8];
  v11 = *(_QWORD *)&byte_444C4A0[v9];
  v19 = v10;
  v18 = v11;
  v12 = sub_CA1930(&v18);
  if ( (_WORD)v16 )
  {
    if ( (_WORD)v16 != 1 && (unsigned __int16)(v16 - 504) > 7u )
    {
      v14 = 16LL * ((unsigned __int16)v16 - 1);
      v13 = *(_QWORD *)&byte_444C4A0[v14];
      LOBYTE(v14) = byte_444C4A0[v14 + 8];
      goto LABEL_18;
    }
LABEL_33:
    BUG();
  }
  v13 = sub_3007260(&v16);
  v22 = v13;
  v23 = v14;
LABEL_18:
  v20 = v13;
  v21 = v14;
  v15 = sub_CA1930(&v20) - 1;
  if ( v15 )
  {
    _BitScanReverse(&v15, v15);
    if ( v12 < (int)(32 - (v15 ^ 0x1F)) )
      return 7;
  }
  return v7;
}
