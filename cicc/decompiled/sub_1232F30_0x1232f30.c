// Function: sub_1232F30
// Address: 0x1232f30
//
__int64 __fastcall sub_1232F30(__int64 a1, unsigned __int8 **a2, __int64 *a3)
{
  unsigned __int64 v4; // r14
  unsigned int v5; // r12d
  const char *v7; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  unsigned __int8 *v10; // rax
  unsigned __int8 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+18h] [rbp-78h] BYREF
  __int64 v20; // [rsp+20h] [rbp-70h] BYREF
  __int64 v21; // [rsp+28h] [rbp-68h] BYREF
  const char *v22[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v23; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 232);
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v19, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after select condition") )
    return 1;
  if ( (unsigned __int8)sub_122FE20((__int64 **)a1, &v20, a3) )
    return 1;
  if ( (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' after select value") )
    return 1;
  v5 = sub_122FE20((__int64 **)a1, &v21, a3);
  if ( (_BYTE)v5 )
    return 1;
  v7 = sub_B489D0(v19, v20, v21);
  v23 = 257;
  if ( v7 )
  {
    if ( *v7 )
    {
      v22[0] = v7;
      LOBYTE(v23) = 3;
    }
    sub_11FD800(a1 + 176, v4, (__int64)v22, 1);
    return 1;
  }
  v8 = v21;
  v9 = v20;
  v18 = v19;
  v10 = (unsigned __int8 *)sub_BD2C40(72, 3u);
  v11 = v10;
  if ( v10 )
  {
    sub_B44260((__int64)v10, *(_QWORD *)(v9 + 8), 57, 3u, 0, 0);
    if ( *((_QWORD *)v11 - 12) )
    {
      v12 = *((_QWORD *)v11 - 11);
      **((_QWORD **)v11 - 10) = v12;
      if ( v12 )
        *(_QWORD *)(v12 + 16) = *((_QWORD *)v11 - 10);
    }
    *((_QWORD *)v11 - 12) = v18;
    if ( v18 )
    {
      v13 = *(_QWORD *)(v18 + 16);
      *((_QWORD *)v11 - 11) = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = v11 - 88;
      *((_QWORD *)v11 - 10) = v18 + 16;
      *(_QWORD *)(v18 + 16) = v11 - 96;
    }
    if ( *((_QWORD *)v11 - 8) )
    {
      v14 = *((_QWORD *)v11 - 7);
      **((_QWORD **)v11 - 6) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = *((_QWORD *)v11 - 6);
    }
    *((_QWORD *)v11 - 8) = v9;
    v15 = *(_QWORD *)(v9 + 16);
    *((_QWORD *)v11 - 7) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = v11 - 56;
    *((_QWORD *)v11 - 6) = v9 + 16;
    *(_QWORD *)(v9 + 16) = v11 - 64;
    if ( *((_QWORD *)v11 - 4) )
    {
      v16 = *((_QWORD *)v11 - 3);
      **((_QWORD **)v11 - 2) = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *((_QWORD *)v11 - 2);
    }
    *((_QWORD *)v11 - 4) = v8;
    if ( v8 )
    {
      v17 = *(_QWORD *)(v8 + 16);
      *((_QWORD *)v11 - 3) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = v11 - 24;
      *((_QWORD *)v11 - 2) = v8 + 16;
      *(_QWORD *)(v8 + 16) = v11 - 32;
    }
    sub_BD6B50(v11, v22);
  }
  *a2 = v11;
  return v5;
}
