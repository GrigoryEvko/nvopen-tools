// Function: sub_11ECEF0
// Address: 0x11ecef0
//
__int64 __fastcall sub_11ECEF0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rdx
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 result; // rax
  unsigned __int64 v9; // r10
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int64 *v19; // [rsp+10h] [rbp-90h]
  __int64 v20; // [rsp+10h] [rbp-90h]
  __int64 v21; // [rsp+18h] [rbp-88h]
  __int64 v22; // [rsp+20h] [rbp-80h]
  __int64 v24; // [rsp+30h] [rbp-70h]
  __int64 v25; // [rsp+38h] [rbp-68h] BYREF
  __int64 v26[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v27; // [rsp+60h] [rbp-40h]

  v22 = sub_B43CC0(a2);
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(_QWORD *)(a2 - 32 * v5);
  v7 = *(_QWORD *)(a2 + 32 * (1 - v5));
  v21 = *(_QWORD *)(a2 + 32 * (2 - v5));
  if ( a4 != 144 )
  {
    BYTE4(v26[0]) = 0;
    BYTE4(v24) = 0;
    v25 = 0x100000001LL;
    if ( sub_11EC990(a1, a2, 2u, v24, 0x100000001LL, v26[0]) )
    {
      if ( a4 != 147 )
      {
LABEL_5:
        result = sub_11CA2E0(v6, v7, a3, *(__int64 **)a1);
        if ( result )
          goto LABEL_6;
        return 0;
      }
      result = sub_11CA290(v6, v7, a3, *(__int64 **)a1);
LABEL_12:
      if ( result )
      {
LABEL_6:
        if ( *(_BYTE *)result == 85 )
          *(_WORD *)(result + 2) = *(_WORD *)(result + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
        return result;
      }
      return 0;
    }
LABEL_14:
    if ( *(_BYTE *)(a1 + 8) )
      return 0;
    v9 = sub_98B430(v7, 8u);
    if ( !v9 )
      return 0;
    v18 = v9;
    LODWORD(v26[0]) = 1;
    sub_11DA2E0(a2, (unsigned int *)v26, 1, v9);
    v19 = *(__int64 **)a1;
    v10 = sub_B43CA0(a2);
    LODWORD(v19) = sub_97FA80(*v19, v10);
    v11 = (_QWORD *)sub_BD5C60(a2);
    v20 = sub_BCCE00(v11, (unsigned int)v19);
    v12 = sub_AD64C0(v20, v18, 0);
    result = sub_11CA4B0(v6, v7, v12, v21, a3, v22, *(__int64 **)a1);
    if ( !result )
      return 0;
    if ( a4 == 144 )
    {
      v27 = 257;
      v13 = (_BYTE *)sub_AD64C0(v20, v18 - 1, 0);
      v14 = *(_QWORD **)(a3 + 72);
      v25 = (__int64)v13;
      v15 = sub_BCB2B0(v14);
      return sub_921130((unsigned int **)a3, v15, v6, (_BYTE **)&v25, 1, (__int64)v26, 3u);
    }
    goto LABEL_12;
  }
  if ( *(_BYTE *)(a1 + 8) == 1 || v7 != v6 )
  {
    BYTE4(v26[0]) = 0;
    BYTE4(v24) = 0;
    v25 = 0x100000001LL;
    if ( sub_11EC990(a1, a2, 2u, v24, 0x100000001LL, v26[0]) )
      goto LABEL_5;
    goto LABEL_14;
  }
  result = sub_11CA050(v7, a3, v22, *(__int64 **)a1);
  v25 = result;
  if ( result )
  {
    v16 = *(_QWORD **)(a3 + 72);
    v27 = 257;
    v17 = sub_BCB2B0(v16);
    return sub_921130((unsigned int **)a3, v17, v6, (_BYTE **)&v25, 1, (__int64)v26, 3u);
  }
  return result;
}
