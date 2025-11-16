// Function: sub_667B60
// Address: 0x667b60
//
__int64 __fastcall sub_667B60(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v4; // bl
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rdi
  int v14; // r13d
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  char v19; // al
  __int64 v20; // rax
  int v21; // eax
  char v22; // dl
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rax
  __int64 v26; // r12
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-58h]
  int v30; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v31; // [rsp+18h] [rbp-48h] BYREF
  __int64 v32; // [rsp+20h] [rbp-40h] BYREF
  _QWORD v33[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a1;
  sub_7B8B50(a1, a2, a3, a4);
  v5 = 27;
  if ( !(unsigned int)sub_7BE280(27, 125, 0, 0) )
    return sub_72C930(v5);
  v32 = sub_724DC0(27, 125, v6, v7, v8, v9);
  v33[0] = *(_QWORD *)&dword_4F063F8;
  v11 = qword_4F061C8;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_BYTE *)(v11 + 75);
  sub_65CD60(&v31);
  v12 = sub_8D2930(v31);
  v13 = v31;
  if ( v12 )
    goto LABEL_4;
  v21 = sub_8D2AC0(v31);
  v13 = v31;
  v14 = v21;
  if ( v21 )
    goto LABEL_4;
  v22 = *(_BYTE *)(v31 + 140);
  if ( v22 == 12 )
  {
    v23 = v31;
    do
    {
      v23 = *(_QWORD *)(v23 + 160);
      v22 = *(_BYTE *)(v23 + 140);
    }
    while ( v22 == 12 );
  }
  if ( v22 == 18 )
  {
LABEL_4:
    if ( v4 == 2 )
    {
      if ( !(unsigned int)sub_8D2C40(v13) )
      {
        v14 = 1;
        sub_685360(3409, v33);
        v13 = v31;
        goto LABEL_8;
      }
    }
    else
    {
      if ( v4 != 3 )
      {
LABEL_6:
        v14 = 0;
LABEL_8:
        while ( *(_BYTE *)(v13 + 140) == 12 )
          v13 = *(_QWORD *)(v13 + 160);
        v15 = *(_QWORD *)(v13 + 128);
        goto LABEL_10;
      }
      if ( !(unsigned int)sub_8D2CC0(v13) )
      {
        v14 = 1;
        sub_685360(3410, v33);
        v13 = v31;
        goto LABEL_8;
      }
    }
    v13 = v31;
    goto LABEL_6;
  }
  v15 = 1;
  if ( !(unsigned int)sub_8D3D40(v31) )
  {
    v24 = *(_BYTE *)(v31 + 140);
    if ( v24 == 12 )
    {
      v25 = v31;
      do
      {
        v25 = *(_QWORD *)(v25 + 160);
        v24 = *(_BYTE *)(v25 + 140);
      }
      while ( v24 == 12 );
    }
    v15 = 0;
    v14 = 1;
    if ( v24 )
      sub_685360(2788, v33);
  }
LABEL_10:
  sub_7BE280(67, 253, 0, 0);
  v33[0] = *(_QWORD *)&dword_4F063F8;
  sub_6BA680(v32);
  v19 = *(_BYTE *)(v32 + 173);
  if ( !v19 )
  {
LABEL_14:
    sub_7BE280(28, 18, 0, 0);
    v20 = qword_4F061C8;
    v5 = (__int64)&v32;
    --*(_BYTE *)(qword_4F061C8 + 75LL);
    --*(_BYTE *)(v20 + 36);
    sub_724E30(&v32);
    return sub_72C930(v5);
  }
  if ( v19 != 12 )
  {
    if ( v19 != 1 )
    {
      sub_6851C0(661, v33);
      goto LABEL_14;
    }
    if ( v14 )
      goto LABEL_14;
    v26 = sub_620FD0(v32, &v30);
    if ( (unsigned __int8)(v4 - 2) <= 1u )
    {
      if ( v30 || (v27 = v26 * v15, ((v27 - 8) & 0xFFFFFFFFFFFFFFF7LL) != 0) )
      {
        v14 = 1;
        sub_6851C0(3413, v33);
        goto LABEL_35;
      }
    }
    else
    {
      if ( v30 )
      {
LABEL_42:
        v14 = 1;
        sub_6851C0(2787, v33);
LABEL_35:
        v29 = 0;
        goto LABEL_36;
      }
      v27 = v26 * v15;
    }
    if ( unk_4F06988 >= v27 )
    {
      if ( (v26 & (v26 - 1)) != 0 )
      {
        v14 = 1;
        sub_6851C0(1685, v33);
      }
      goto LABEL_35;
    }
    goto LABEL_42;
  }
  v26 = 1;
  v29 = sub_724E50(&v32, 253, v16, v17, v18);
LABEL_36:
  sub_7BE280(28, 18, 0, 0);
  v28 = qword_4F061C8;
  v5 = (__int64)&v32;
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  --*(_BYTE *)(v28 + 36);
  sub_724E30(&v32);
  if ( v14 )
    return sub_72C930(v5);
  result = sub_72B5A0(v31, v26, v4);
  *(_QWORD *)(result + 168) = v29;
  return result;
}
