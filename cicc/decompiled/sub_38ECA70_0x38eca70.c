// Function: sub_38ECA70
// Address: 0x38eca70
//
__int64 __fastcall sub_38ECA70(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v5; // r14
  unsigned int v6; // r12d
  __int64 v8; // rax
  __int64 v9; // r13
  unsigned int *v10; // rsi
  char v11; // dl
  signed __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // [rsp+18h] [rbp-98h]
  __int64 v17; // [rsp+18h] [rbp-98h]
  _QWORD v18[2]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v19; // [rsp+30h] [rbp-80h] BYREF
  __int64 v20; // [rsp+38h] [rbp-78h] BYREF
  char *v21; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v22; // [rsp+48h] [rbp-68h]
  __int16 v23; // [rsp+50h] [rbp-60h]
  char **v24; // [rsp+60h] [rbp-50h] BYREF
  char *v25; // [rsp+68h] [rbp-48h]
  __int16 v26; // [rsp+70h] [rbp-40h]

  v18[0] = a2;
  v18[1] = a3;
  v5 = sub_3909290(a1 + 144);
  if ( !*(_BYTE *)(a1 + 845) && (unsigned __int8)sub_38E36C0(a1) )
    return 1;
  v6 = sub_38EB9C0(a1, &v19);
  if ( (_BYTE)v6 )
    return 1;
  if ( v19 < 0 )
  {
    v21 = "'";
    v22 = v18;
    v23 = 1283;
    v24 = &v21;
    v26 = 770;
    v25 = "' directive with negative repeat count has no effect";
    sub_38E4170((_QWORD *)a1, v5, (__int64)&v24, 0, 0);
    return v6;
  }
  v21 = "unexpected token in '";
  v22 = v18;
  v24 = &v21;
  v26 = 770;
  v23 = 1283;
  v25 = "' directive";
  if ( (unsigned __int8)sub_3909E20(a1, 25, &v24) )
    return 1;
  v8 = sub_3909290(a1 + 144);
  v24 = 0;
  v9 = v8;
  if ( sub_38EB6A0(a1, &v20, (__int64)&v24) )
  {
    return 1;
  }
  else
  {
    v10 = (unsigned int *)v20;
    if ( *(_DWORD *)v20 == 1 )
    {
      v11 = 8 * a4;
      v12 = *(_QWORD *)(v20 + 16);
      if ( v12 > 0xFFFFFFFFFFFFFFFFLL >> (64 - 8 * (unsigned __int8)a4)
        && (v12 < -(1LL << (v11 - 1)) || v12 > (1LL << (v11 - 1)) - 1) )
      {
        v24 = (char **)"literal value out of range for directive";
        v26 = 259;
        return (unsigned int)sub_3909790(a1, v9, &v24, 0, 0);
      }
      v16 = v19;
      if ( v19 )
      {
        v13 = 0;
        do
        {
          ++v13;
          (*(void (__fastcall **)(_QWORD, signed __int64, _QWORD))(**(_QWORD **)(a1 + 328) + 424LL))(
            *(_QWORD *)(a1 + 328),
            v12,
            a4);
        }
        while ( v16 != v13 );
      }
    }
    else
    {
      v14 = v19;
      if ( v19 )
      {
        v15 = 0;
        while ( 1 )
        {
          v17 = v15;
          sub_38DDD30(*(_QWORD *)(a1 + 328), v10);
          v15 = v17 + 1;
          if ( v14 == v17 + 1 )
            break;
          v10 = (unsigned int *)v20;
        }
      }
    }
    v21 = "unexpected token in '";
    v26 = 770;
    v22 = v18;
    v23 = 1283;
    v24 = &v21;
    v25 = "' directive";
    return (unsigned int)sub_3909E20(a1, 9, &v24);
  }
}
