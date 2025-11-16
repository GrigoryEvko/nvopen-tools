// Function: sub_15641F0
// Address: 0x15641f0
//
_QWORD *__fastcall sub_15641F0(__int64 *a1, int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v8; // rax
  _QWORD *v9; // r14
  __int64 v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rax
  int v13; // esi
  __int64 v14; // rdi
  unsigned __int64 *v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rsi
  __int64 v21; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+28h] [rbp-58h] BYREF
  char v25[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v26; // [rsp+40h] [rbp-40h]

  v26 = 257;
  v8 = sub_1648A60(56, 2);
  v9 = (_QWORD *)v8;
  if ( v8 )
  {
    v10 = v8;
    v11 = *(_QWORD **)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v21 = v11[4];
      v12 = sub_1643320(*v11);
      v13 = sub_16463B0(v12, v21);
    }
    else
    {
      v13 = sub_1643320(*v11);
    }
    sub_15FEC10((_DWORD)v9, v13, 51, a2, a3, a4, (__int64)v25, 0);
  }
  else
  {
    v10 = 0;
  }
  v14 = a1[1];
  if ( v14 )
  {
    v15 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v14 + 40, v9);
    v16 = v9[3];
    v17 = *v15;
    v9[4] = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[3] = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v9 + 3;
    *v15 = *v15 & 7 | (unsigned __int64)(v9 + 3);
  }
  sub_164B780(v10, a5);
  v18 = *a1;
  if ( *a1 )
  {
    v24 = *a1;
    sub_1623A60(&v24, v18, 2);
    if ( v9[6] )
      sub_161E7C0(v9 + 6);
    v19 = v24;
    v9[6] = v24;
    if ( v19 )
      sub_1623210(&v24, v19, v9 + 6);
  }
  return v9;
}
