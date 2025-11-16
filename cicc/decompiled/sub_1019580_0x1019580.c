// Function: sub_1019580
// Address: 0x1019580
//
__int64 __fastcall sub_1019580(unsigned __int8 *a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // rax
  int v12; // ebx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  unsigned __int8 *v16; // r14
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 *v21; // rax
  int v22; // ecx
  unsigned __int8 **v23; // rdx
  _BYTE *v24; // rsi
  __int64 v25; // r14
  __int64 v27; // [rsp+0h] [rbp-70h]
  unsigned __int8 **v29; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+18h] [rbp-58h]
  _BYTE v31[80]; // [rsp+20h] [rbp-50h] BYREF

  v6 = *a1;
  if ( v6 == 40 )
  {
    v7 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v7 = -32;
    if ( v6 != 85 )
    {
      v7 = -96;
      if ( v6 != 34 )
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v8 = sub_BD2BC0((__int64)a1);
    v10 = v8 + v9;
    v11 = 0;
    if ( (a1[7] & 0x80u) != 0 )
      v11 = sub_BD2BC0((__int64)a1);
    if ( (unsigned int)((v10 - v11) >> 4) )
    {
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v12 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
      if ( (a1[7] & 0x80u) == 0 )
        BUG();
      v13 = sub_BD2BC0((__int64)a1);
      v7 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
    }
  }
  v15 = *((_DWORD *)a1 + 1);
  v16 = &a1[v7];
  v29 = (unsigned __int8 **)v31;
  v30 = 0x400000000LL;
  v17 = 32LL * (v15 & 0x7FFFFFF);
  v18 = (__int64 *)&a1[-v17];
  v19 = v7 + v17;
  v20 = v19 >> 5;
  if ( (unsigned __int64)v19 > 0x80 )
  {
    v27 = v19 >> 5;
    sub_C8D5F0((__int64)&v29, v31, v19 >> 5, 8u, v20, a6);
    v23 = v29;
    v22 = v30;
    LODWORD(v20) = v27;
    v21 = (__int64 *)&v29[(unsigned int)v30];
  }
  else
  {
    v21 = (__int64 *)v31;
    v22 = 0;
    v23 = (unsigned __int8 **)v31;
  }
  if ( v18 != (__int64 *)v16 )
  {
    do
    {
      if ( v21 )
        *v21 = *v18;
      v18 += 4;
      ++v21;
    }
    while ( v16 != (unsigned __int8 *)v18 );
    v23 = v29;
    v22 = v30;
  }
  v24 = (_BYTE *)*((_QWORD *)a1 - 4);
  LODWORD(v30) = v20 + v22;
  v25 = sub_FFF020((__int64)a1, v24, v23, (unsigned int)(v20 + v22), (__int64)a2);
  if ( !v25 )
  {
    v24 = (_BYTE *)*((_QWORD *)a1 - 4);
    v25 = (__int64)sub_1018220((__int64)a1, (__int64)v24, (__int64 *)v29, (unsigned int)v30, a2);
  }
  if ( v29 != (unsigned __int8 **)v31 )
    _libc_free(v29, v24);
  return v25;
}
