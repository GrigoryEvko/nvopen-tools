// Function: sub_38AE200
// Address: 0x38ae200
//
__int64 __fastcall sub_38AE200(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v7; // r14
  unsigned int v8; // r12d
  const char *v10; // rax
  _QWORD *v11; // r14
  __int64 *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rbx
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // [rsp+0h] [rbp-80h]
  _QWORD *v25; // [rsp+8h] [rbp-78h]
  __int64 *v26; // [rsp+18h] [rbp-68h] BYREF
  __int64 *v27; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v28; // [rsp+28h] [rbp-58h] BYREF
  __int64 v29[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v30; // [rsp+40h] [rbp-40h]

  v7 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v26, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after select condition") )
    return 1;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v27, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after select value") )
    return 1;
  v8 = sub_38AB270((__int64 **)a1, &v28, a3, a4, a5, a6);
  if ( (_BYTE)v8 )
  {
    return 1;
  }
  else
  {
    v10 = sub_15F50B0(v26, (__int64)v27, v28);
    v30 = 257;
    if ( v10 )
    {
      if ( *v10 )
      {
        v29[0] = (__int64)v10;
        LOBYTE(v30) = 3;
      }
      return (unsigned int)sub_38814C0(a1 + 8, v7, (__int64)v29);
    }
    else
    {
      v11 = v28;
      v12 = v27;
      v24 = v26;
      v13 = sub_1648A60(56, 3u);
      v14 = (__int64)v13;
      if ( v13 )
      {
        v25 = v13 - 9;
        sub_15F1EA0((__int64)v13, *v12, 55, (__int64)(v13 - 9), 3, 0);
        if ( *(_QWORD *)(v14 - 72) )
        {
          v15 = *(_QWORD *)(v14 - 64);
          v16 = *(_QWORD *)(v14 - 56) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v16 = v15;
          if ( v15 )
            *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
        }
        *(_QWORD *)(v14 - 72) = v24;
        if ( v24 )
        {
          v17 = v24[1];
          *(_QWORD *)(v14 - 64) = v17;
          if ( v17 )
            *(_QWORD *)(v17 + 16) = (v14 - 64) | *(_QWORD *)(v17 + 16) & 3LL;
          *(_QWORD *)(v14 - 56) = *(_QWORD *)(v14 - 56) & 3LL | (unsigned __int64)(v24 + 1);
          v24[1] = (__int64)v25;
        }
        if ( *(_QWORD *)(v14 - 48) )
        {
          v18 = *(_QWORD *)(v14 - 40);
          v19 = *(_QWORD *)(v14 - 32) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v19 = v18;
          if ( v18 )
            *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
        }
        *(_QWORD *)(v14 - 48) = v12;
        v20 = v12[1];
        *(_QWORD *)(v14 - 40) = v20;
        if ( v20 )
          *(_QWORD *)(v20 + 16) = (v14 - 40) | *(_QWORD *)(v20 + 16) & 3LL;
        *(_QWORD *)(v14 - 32) = (unsigned __int64)(v12 + 1) | *(_QWORD *)(v14 - 32) & 3LL;
        v12[1] = v14 - 48;
        if ( *(_QWORD *)(v14 - 24) )
        {
          v21 = *(_QWORD *)(v14 - 16);
          v22 = *(_QWORD *)(v14 - 8) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v22 = v21;
          if ( v21 )
            *(_QWORD *)(v21 + 16) = *(_QWORD *)(v21 + 16) & 3LL | v22;
        }
        *(_QWORD *)(v14 - 24) = v11;
        if ( v11 )
        {
          v23 = v11[1];
          *(_QWORD *)(v14 - 16) = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = (v14 - 16) | *(_QWORD *)(v23 + 16) & 3LL;
          *(_QWORD *)(v14 - 8) = *(_QWORD *)(v14 - 8) & 3LL | (unsigned __int64)(v11 + 1);
          v11[1] = v14 - 24;
        }
        sub_164B780(v14, v29);
      }
      *a2 = v14;
    }
  }
  return v8;
}
