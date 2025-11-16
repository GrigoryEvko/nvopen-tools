// Function: sub_13CBA60
// Address: 0x13cba60
//
__int64 __fastcall sub_13CBA60(__int64 a1, __int64 *a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  int v7; // eax
  _QWORD *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // r10
  int v12; // r10d
  char v13; // al
  __int64 v14; // rbx
  __int64 v15; // r9
  char v16; // al
  __int64 v17; // r8
  char v18; // al
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // [rsp+0h] [rbp-50h]
  __int64 v24; // [rsp+0h] [rbp-50h]
  int v25; // [rsp+8h] [rbp-48h]
  int v26; // [rsp+8h] [rbp-48h]
  int v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+10h] [rbp-40h]
  int v30; // [rsp+10h] [rbp-40h]
  _QWORD *v31; // [rsp+18h] [rbp-38h]
  _QWORD *v32; // [rsp+18h] [rbp-38h]
  int v33; // [rsp+18h] [rbp-38h]

  v7 = *((unsigned __int8 *)a2 + 16);
  if ( (unsigned __int8)v7 <= 0x10u )
    return sub_14D7A60(a1, a2, a3, *a4);
  if ( (unsigned __int8)v7 <= 0x17u )
    goto LABEL_6;
  if ( (unsigned int)(v7 - 60) > 0xC )
    goto LABEL_6;
  v9 = (_QWORD *)*(a2 - 3);
  v10 = *v9;
  if ( a3 != *v9 )
    goto LABEL_6;
  v12 = v7 - 24;
  v13 = *(_BYTE *)(a3 + 8);
  if ( v13 == 16 )
    v13 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  v14 = *a2;
  v15 = 0;
  if ( v13 == 15 )
  {
    v26 = v12;
    v29 = *v9;
    v32 = a4;
    v21 = sub_15A9650(*a4, *v9, v10, a4, a5, 0);
    v12 = v26;
    v10 = v29;
    a4 = v32;
    v15 = v21;
  }
  v16 = *(_BYTE *)(v14 + 8);
  if ( v16 == 16 )
    v16 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
  v17 = 0;
  if ( v16 == 15 )
  {
    v23 = v15;
    v25 = v12;
    v28 = v10;
    v31 = a4;
    v20 = sub_15A9650(*a4, v14, v10, a4, 0, v15);
    v15 = v23;
    v12 = v25;
    v10 = v28;
    a4 = v31;
    v17 = v20;
  }
  v18 = *(_BYTE *)(v10 + 8);
  if ( v18 == 16 )
    v18 = *(_BYTE *)(**(_QWORD **)(v10 + 16) + 8LL);
  v19 = 0;
  if ( v18 == 15 )
  {
    v24 = v17;
    v27 = v15;
    v30 = v12;
    v33 = v10;
    v22 = sub_15A9650(*a4, v10, v10, a4, v17, v15);
    v17 = v24;
    LODWORD(v15) = v27;
    v12 = v30;
    LODWORD(v10) = v33;
    v19 = v22;
  }
  if ( (unsigned int)sub_15FB960(v12, a1, v10, v14, v10, v15, v17, v19) != 47 )
  {
LABEL_6:
    v9 = 0;
    if ( (_DWORD)a1 == 47 )
    {
      v11 = 0;
      if ( a3 == *a2 )
        return (__int64)a2;
      return v11;
    }
  }
  return (__int64)v9;
}
