// Function: sub_1522F40
// Address: 0x1522f40
//
_QWORD *__fastcall sub_1522F40(__int64 *a1, unsigned int a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r15
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  _QWORD *v9; // r12
  unsigned __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rbx
  _QWORD *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  _BYTE v18[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v19; // [rsp+20h] [rbp-40h]

  v3 = a2;
  if ( a2 == -1 )
    return 0;
  v4 = a1[1];
  v6 = *a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - *a1) >> 3);
  if ( a2 >= (unsigned int)v7 )
  {
    v11 = a2 + 1;
    if ( v11 > v7 )
    {
      sub_14EF7A0((__int64)a1, v11 - v7);
      v6 = *a1;
    }
    else if ( v11 < v7 )
    {
      v17 = v6 + 24 * v11;
      if ( v4 != v17 )
      {
        v12 = v6 + 24 * v11;
        do
        {
          v13 = *(_QWORD *)(v12 + 16);
          if ( v13 != 0 && v13 != -8 && v13 != -16 )
            sub_1649B30(v12);
          v12 += 24;
        }
        while ( v4 != v12 );
        v6 = *a1;
        a1[1] = v17;
      }
    }
  }
  v8 = 24 * v3;
  v9 = *(_QWORD **)(v6 + v8 + 16);
  if ( v9 )
  {
    if ( !a3 || a3 == *v9 )
      return v9;
    return 0;
  }
  if ( !a3 )
    return 0;
  v19 = 257;
  v14 = sub_22077B0(40);
  v9 = (_QWORD *)v14;
  if ( v14 )
    sub_15E0280(v14, a3, v18, 0, 0);
  v15 = *a1 + v8;
  v16 = *(_QWORD **)(v15 + 16);
  if ( v9 != v16 )
  {
    if ( v16 + 1 != 0 && v16 != 0 && v16 != (_QWORD *)-16LL )
      sub_1649B30(v15);
    *(_QWORD *)(v15 + 16) = v9;
    if ( !v9 )
      return 0;
    if ( v9 != (_QWORD *)-8LL && v9 != (_QWORD *)-16LL )
      sub_164C220(v15);
  }
  return v9;
}
