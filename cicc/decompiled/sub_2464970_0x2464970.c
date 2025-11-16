// Function: sub_2464970
// Address: 0x2464970
//
unsigned __int64 __fastcall sub_2464970(__int64 *a1, unsigned int **a2, unsigned __int64 a3, __int64 a4, char a5)
{
  unsigned __int64 result; // rax
  __int64 v6; // r14
  unsigned __int64 v10; // rsi
  __int64 v11; // r11
  __int64 v12; // rsi
  _BYTE *v13; // rax
  __int64 v14; // rdx
  unsigned __int8 v15; // al
  int v16; // edi
  __int64 *v17; // r14
  __int64 **v18; // rax
  __int64 v19; // rax
  __int64 **v20; // r14
  unsigned int v21; // esi
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  unsigned int v24; // [rsp+8h] [rbp-88h]
  unsigned __int64 v25; // [rsp+8h] [rbp-88h]
  unsigned __int64 v28; // [rsp+18h] [rbp-78h]
  int v29; // [rsp+28h] [rbp-68h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  __int64 v31; // [rsp+38h] [rbp-58h]
  __int16 v32; // [rsp+50h] [rbp-40h]

  result = a3;
  v6 = *(_QWORD *)(a3 + 8);
  if ( a4 == v6 )
    return result;
  if ( (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17 > 1 )
  {
    v30 = sub_BCAE30(v6);
    v31 = v14;
    v10 = (unsigned int)sub_CA1930(&v30);
  }
  else
  {
    v10 = *(_DWORD *)(v6 + 32) * (unsigned int)sub_BCB060(v6);
  }
  if ( (unsigned int)*(unsigned __int8 *)(a4 + 8) - 17 > 1 )
  {
    v30 = sub_BCAE30(a4);
    v31 = v23;
    v11 = (unsigned int)sub_CA1930(&v30);
  }
  else
  {
    v11 = *(_DWORD *)(a4 + 32) * (unsigned int)sub_BCB060(a4);
  }
  if ( v10 > 1 && v11 == 1 )
  {
    v12 = *(_QWORD *)(a3 + 8);
    v32 = 257;
    v13 = (_BYTE *)sub_24637B0(a1, v12);
    return sub_92B530(a2, 0x21u, a3, v13, (__int64)&v30);
  }
  v15 = *(_BYTE *)(a4 + 8);
  if ( v15 == 12 )
  {
    if ( *(_BYTE *)(v6 + 8) != 12 )
      goto LABEL_16;
LABEL_21:
    v32 = 257;
    return sub_921630(a2, a3, a4, a5, (__int64)&v30);
  }
  if ( (unsigned int)v15 - 17 <= 1 )
  {
    v16 = *(unsigned __int8 *)(v6 + 8);
    if ( (unsigned int)(v16 - 17) <= 1
      && (v15 == 18) == ((_BYTE)v16 == 18)
      && *(_DWORD *)(a4 + 32) == *(_DWORD *)(v6 + 32) )
    {
      goto LABEL_21;
    }
  }
LABEL_16:
  v17 = a1;
  v24 = v11;
  v32 = 257;
  v18 = (__int64 **)sub_BCD140(*(_QWORD **)(a1[1] + 72), v10);
  v28 = sub_24633A0((__int64 *)a2, 0x31u, a3, v18, (__int64)&v30, 0, v29, 0);
  v19 = v17[1];
  v32 = 257;
  v20 = (__int64 **)sub_BCD140(*(_QWORD **)(v19 + 72), v24);
  v25 = v28;
  LODWORD(v28) = sub_BCB060(*(_QWORD *)(v28 + 8));
  v21 = 39 - ((a5 == 0) - 1);
  if ( (unsigned int)v28 > (unsigned int)sub_BCB060((__int64)v20) )
    v21 = 38;
  v22 = sub_24633A0((__int64 *)a2, v21, v25, v20, (__int64)&v30, 0, v29, 0);
  v32 = 257;
  return sub_24633A0((__int64 *)a2, 0x31u, v22, (__int64 **)a4, (__int64)&v30, 0, v29, 0);
}
