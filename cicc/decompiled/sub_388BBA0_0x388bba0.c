// Function: sub_388BBA0
// Address: 0x388bba0
//
__int64 __fastcall sub_388BBA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  unsigned int v5; // r15d
  unsigned __int64 v6; // rsi
  int v9; // r12d
  int v10; // eax
  unsigned __int64 v11; // rsi
  unsigned int v12; // eax
  unsigned __int64 v13; // rsi
  int v14; // eax
  int v15; // [rsp+4h] [rbp-8Ch]
  _QWORD v16[2]; // [rsp+10h] [rbp-80h] BYREF
  char *v17; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v18; // [rsp+28h] [rbp-68h]
  __int16 v19; // [rsp+30h] [rbp-60h]
  char **v20; // [rsp+40h] [rbp-50h] BYREF
  char *v21; // [rsp+48h] [rbp-48h]
  __int16 v22; // [rsp+50h] [rbp-40h]

  v4 = a1 + 8;
  v5 = *(unsigned __int8 *)(a4 + 4);
  v16[0] = a2;
  v16[1] = a3;
  if ( !(_BYTE)v5 )
  {
    v9 = 0;
    v10 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v10;
    while ( v10 != 390 )
    {
      if ( v10 != 385 )
        goto LABEL_10;
      v15 = sub_15AFD40(*(_QWORD *)(a1 + 72), *(_QWORD *)(a1 + 80));
      if ( v15 )
      {
        v14 = sub_3887100(v4);
        *(_DWORD *)(a1 + 64) = v14;
        goto LABEL_13;
      }
      v11 = *(_QWORD *)(a1 + 56);
      v17 = "invalid debug info flag flag '";
      v18 = (_QWORD *)(a1 + 72);
      v19 = 1027;
      v22 = 770;
      v20 = &v17;
      v21 = "'";
      v12 = sub_38814C0(v4, v11, (__int64)&v20);
LABEL_11:
      if ( (_BYTE)v12 )
        return v12;
      v14 = *(_DWORD *)(a1 + 64);
LABEL_13:
      v9 |= v15;
      if ( v14 != 15 )
      {
        *(_BYTE *)(a4 + 4) = 1;
        *(_DWORD *)a4 = v9;
        return v5;
      }
      v10 = sub_3887100(v4);
      *(_DWORD *)(a1 + 64) = v10;
    }
    if ( *(_BYTE *)(a1 + 164) )
    {
      LODWORD(v20) = v15;
      v12 = sub_388BA90(a1, &v20);
      v15 = (int)v20;
      goto LABEL_11;
    }
LABEL_10:
    v13 = *(_QWORD *)(a1 + 56);
    v22 = 259;
    v20 = (char **)"expected debug info flag";
    v12 = sub_38814C0(v4, v13, (__int64)&v20);
    goto LABEL_11;
  }
  v17 = "field '";
  v18 = v16;
  v22 = 770;
  v6 = *(_QWORD *)(a1 + 56);
  v20 = &v17;
  v19 = 1283;
  v21 = "' cannot be specified more than once";
  return (unsigned int)sub_38814C0(a1 + 8, v6, (__int64)&v20);
}
