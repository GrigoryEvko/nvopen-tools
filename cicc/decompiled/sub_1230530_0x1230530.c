// Function: sub_1230530
// Address: 0x1230530
//
__int64 __fastcall sub_1230530(__int64 a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 v5; // r14
  unsigned int v6; // r12d
  bool v8; // zf
  __int64 v9; // rsi
  __int64 v10; // r8
  const char *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r10
  unsigned __int64 *v14; // r9
  unsigned __int64 v15; // rdx
  __int64 v16; // r10
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // r11
  unsigned __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 *v24; // r15
  unsigned __int64 *v25; // [rsp+0h] [rbp-100h]
  __int64 v26; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v27; // [rsp+10h] [rbp-F0h]
  __int64 v28; // [rsp+10h] [rbp-F0h]
  __int64 v29; // [rsp+10h] [rbp-F0h]
  int v30; // [rsp+18h] [rbp-E8h]
  __int64 *v31; // [rsp+18h] [rbp-E8h]
  __int64 v32; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v33; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v34; // [rsp+38h] [rbp-C8h] BYREF
  const char *v35; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+48h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+50h] [rbp-B0h] BYREF

  v5 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v32, a3)
    || (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after indirectbr address")
    || (unsigned __int8)sub_120AFE0(a1, 6, "expected '[' with indirectbr") )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)(v32 + 8) + 8LL) != 14 )
  {
    v37[17] = 1;
    v35 = "indirectbr address must have pointer type";
    v37[16] = 3;
    sub_11FD800(a1 + 176, v5, (__int64)&v35, 1);
    return 1;
  }
  v8 = *(_DWORD *)(a1 + 240) == 7;
  v35 = v37;
  v36 = 0x1000000000LL;
  if ( v8 )
    goto LABEL_21;
  v9 = (__int64)&v33;
  v34 = 0;
  if ( !(unsigned __int8)sub_122FEA0(a1, &v33, &v34, a3) )
  {
    v12 = (unsigned int)v36;
    v13 = v33;
    v14 = &v34;
    v15 = (unsigned int)v36 + 1LL;
    if ( v15 > HIDWORD(v36) )
    {
      v28 = v33;
      sub_C8D5F0((__int64)&v35, v37, v15, 8u, v10, (__int64)&v34);
      v12 = (unsigned int)v36;
      v14 = &v34;
      v13 = v28;
    }
    *(_QWORD *)&v35[8 * v12] = v13;
    LODWORD(v36) = v36 + 1;
    if ( *(_DWORD *)(a1 + 240) == 4 )
    {
      v16 = a1 + 176;
      do
      {
        v27 = v14;
        v26 = v16;
        v9 = (__int64)&v33;
        *(_DWORD *)(a1 + 240) = sub_1205200(v16);
        v34 = 0;
        if ( (unsigned __int8)sub_122FEA0(a1, &v33, v27, a3) )
          goto LABEL_10;
        v18 = (unsigned int)v36;
        v19 = v33;
        v14 = v27;
        v20 = (unsigned int)v36 + 1LL;
        v16 = v26;
        if ( v20 > HIDWORD(v36) )
        {
          v25 = v27;
          v29 = v33;
          sub_C8D5F0((__int64)&v35, v37, v20, 8u, v17, (__int64)v14);
          v18 = (unsigned int)v36;
          v14 = v25;
          v16 = v26;
          v19 = v29;
        }
        *(_QWORD *)&v35[8 * v18] = v19;
        LODWORD(v36) = v36 + 1;
      }
      while ( *(_DWORD *)(a1 + 240) == 4 );
    }
LABEL_21:
    v9 = 7;
    v6 = sub_120AFE0(a1, 7, "expected ']' at end of block list");
    if ( (_BYTE)v6 )
    {
      v11 = v35;
    }
    else
    {
      v21 = v32;
      v30 = v36;
      v22 = sub_BD2DA0(80);
      v23 = v22;
      if ( v22 )
      {
        v9 = v21;
        sub_B546A0(v22, v21, v30, 0, 0);
      }
      v11 = v35;
      v31 = (__int64 *)&v35[8 * (unsigned int)v36];
      if ( v31 != (__int64 *)v35 )
      {
        v24 = (__int64 *)v35;
        do
        {
          v9 = *v24++;
          sub_B54850(v23, v9);
        }
        while ( v31 != v24 );
        v11 = v35;
      }
      *a2 = v23;
    }
    goto LABEL_11;
  }
LABEL_10:
  v11 = v35;
  v6 = 1;
LABEL_11:
  if ( v11 != v37 )
    _libc_free(v11, v9);
  return v6;
}
