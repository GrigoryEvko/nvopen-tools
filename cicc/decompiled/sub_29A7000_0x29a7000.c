// Function: sub_29A7000
// Address: 0x29a7000
//
char __fastcall sub_29A7000(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // r14
  unsigned __int8 *v4; // r15
  __int64 v5; // r12
  char result; // al
  __int64 *v7; // r12
  const char *v8; // rax
  __int64 v9; // rcx
  unsigned __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // r12
  __int64 v13; // r12
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // [rsp+8h] [rbp-68h]
  _QWORD v19[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v2 = sub_D4B130(*a1);
  v3 = *(_BYTE **)(a2 + 24);
  v4 = *(unsigned __int8 **)a2;
  v5 = v2;
  result = sub_98ED60(*(unsigned __int8 **)a2, 0, (__int64)v3, a1[2], 0);
  if ( !result )
  {
    v7 = (__int64 *)(v5 + 48);
    v8 = sub_BD5D20((__int64)v4);
    v20 = 773;
    v9 = *v7;
    v19[0] = v8;
    v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
    v19[1] = v11;
    v19[2] = ".frozen";
    if ( (__int64 *)v10 == v7 )
    {
      v12 = 0;
    }
    else
    {
      if ( !v10 )
        BUG();
      v12 = v10 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 >= 0xB )
        v12 = 0;
    }
    v13 = v12 + 24;
    v14 = sub_BD2C40(72, 1u);
    if ( v14 )
    {
      v18 = v14;
      sub_B549F0((__int64)v14, (__int64)v4, (__int64)v19, v13, 0);
      if ( *(_QWORD *)a2 )
      {
        v15 = *(_QWORD *)(a2 + 8);
        **(_QWORD **)(a2 + 16) = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(a2 + 16);
      }
      *(_QWORD *)a2 = v18;
      v16 = v18[2];
      *(_QWORD *)(a2 + 8) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = a2 + 8;
      *(_QWORD *)(a2 + 16) = v18 + 2;
      v18[2] = a2;
    }
    else if ( *(_QWORD *)a2 )
    {
      v17 = *(_QWORD *)(a2 + 8);
      **(_QWORD **)(a2 + 16) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = *(_QWORD *)(a2 + 16);
      *(_QWORD *)a2 = 0;
    }
    return sub_DAC8D0(a1[1], v3);
  }
  return result;
}
