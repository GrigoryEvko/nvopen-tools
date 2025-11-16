// Function: sub_38AB950
// Address: 0x38ab950
//
__int64 __fastcall sub_38AB950(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v8; // r14
  unsigned int v9; // r12d
  bool v11; // zf
  int v12; // r8d
  __int64 v13; // rax
  unsigned __int64 *v14; // r9
  __int64 v15; // r10
  int v16; // r8d
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v27; // [rsp+8h] [rbp-F8h]
  unsigned __int64 *v28; // [rsp+10h] [rbp-F0h]
  __int64 v29; // [rsp+10h] [rbp-F0h]
  int v30; // [rsp+18h] [rbp-E8h]
  __int64 v31; // [rsp+18h] [rbp-E8h]
  __int64 v32; // [rsp+28h] [rbp-D8h] BYREF
  __int64 v33; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v34; // [rsp+38h] [rbp-C8h] BYREF
  const char *v35; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+48h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+50h] [rbp-B0h] BYREF

  v8 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v32, a3, a4, a5, a6)
    || (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after indirectbr address")
    || (unsigned __int8)sub_388AF10(a1, 6, "expected '[' with indirectbr") )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)v32 + 8LL) != 15 )
  {
    v37[1] = 1;
    v35 = "indirectbr address must have pointer type";
    v37[0] = 3;
    return (unsigned int)sub_38814C0(a1 + 8, v8, (__int64)&v35);
  }
  v11 = *(_DWORD *)(a1 + 64) == 7;
  v35 = v37;
  v36 = 0x1000000000LL;
  if ( v11 )
    goto LABEL_21;
  v34 = 0;
  if ( !(unsigned __int8)sub_38AB2F0(a1, &v33, &v34, a3, a4, a5, a6) )
  {
    v13 = (unsigned int)v36;
    v14 = &v34;
    if ( (unsigned int)v36 >= HIDWORD(v36) )
    {
      sub_16CD150((__int64)&v35, v37, 0, 8, v12, (int)&v34);
      v13 = (unsigned int)v36;
      v14 = &v34;
    }
    *(_QWORD *)&v35[8 * v13] = v33;
    LODWORD(v36) = v36 + 1;
    if ( *(_DWORD *)(a1 + 64) == 4 )
    {
      v15 = a1 + 8;
      do
      {
        v28 = v14;
        v26 = v15;
        *(_DWORD *)(a1 + 64) = sub_3887100(v15);
        v34 = 0;
        if ( (unsigned __int8)sub_38AB2F0(a1, &v33, v28, a3, a4, a5, a6) )
          goto LABEL_10;
        v17 = (unsigned int)v36;
        v14 = v28;
        v15 = v26;
        if ( (unsigned int)v36 >= HIDWORD(v36) )
        {
          v27 = v28;
          v29 = v15;
          sub_16CD150((__int64)&v35, v37, 0, 8, v16, (int)v14);
          v17 = (unsigned int)v36;
          v14 = v27;
          v15 = v29;
        }
        *(_QWORD *)&v35[8 * v17] = v33;
        LODWORD(v36) = v36 + 1;
      }
      while ( *(_DWORD *)(a1 + 64) == 4 );
    }
LABEL_21:
    v9 = sub_388AF10(a1, 7, "expected ']' at end of block list");
    if ( !(_BYTE)v9 )
    {
      v18 = v32;
      v30 = v36;
      v19 = sub_1648B60(64);
      v23 = v19;
      if ( v19 )
        sub_1600240(v19, v18, v30, 0);
      v24 = 0;
      v31 = 8LL * (unsigned int)v36;
      if ( (_DWORD)v36 )
      {
        do
        {
          v25 = *(_QWORD *)&v35[v24];
          v24 += 8;
          sub_1600410(v23, v25, (__int64)v35, v20, v21, v22);
        }
        while ( v24 != v31 );
      }
      *a2 = v23;
    }
    goto LABEL_11;
  }
LABEL_10:
  v9 = 1;
LABEL_11:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v9;
}
