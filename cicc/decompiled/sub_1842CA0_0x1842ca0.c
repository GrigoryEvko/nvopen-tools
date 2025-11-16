// Function: sub_1842CA0
// Address: 0x1842ca0
//
__int64 __fastcall sub_1842CA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  char v4; // al
  __int64 v5; // r8
  __int64 v6; // r14
  __int64 i; // r15
  __int64 j; // r15
  unsigned __int64 v9; // rax
  unsigned __int8 v10; // dl
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rbx
  bool v13; // zf
  unsigned __int64 v14; // rax
  __int64 v15; // r13
  __int64 v16; // r12
  unsigned __int64 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rdi
  unsigned __int64 v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // r14
  __int64 v23; // rdx
  int v24; // r8d
  int v25; // r9d
  int v26; // r13d
  __int64 v27; // rax
  _BYTE *v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+18h] [rbp-58h]
  _BYTE v30[80]; // [rsp+20h] [rbp-50h] BYREF

  if ( sub_15E4F60(a2) )
    return 0;
  sub_15E4B50(a2);
  if ( v4 || (*(_BYTE *)(a2 + 32) & 0xFu) - 7 <= 1 && !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
    return 0;
  v2 = sub_1560180(a2 + 112, 18);
  if ( (_BYTE)v2 )
    return 0;
  if ( !*(_QWORD *)(a2 + 8) )
    return v2;
  v28 = v30;
  v29 = 0x800000000LL;
  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, 18);
    v5 = *(_QWORD *)(a2 + 88);
    v6 = v5 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, 18);
      v5 = *(_QWORD *)(a2 + 88);
    }
  }
  else
  {
    v5 = *(_QWORD *)(a2 + 88);
    v6 = v5 + 40LL * *(_QWORD *)(a2 + 96);
  }
  for ( i = v5; v6 != i; i += 40 )
  {
    if ( !(unsigned __int8)sub_15E02D0(i) && !*(_QWORD *)(i + 8) && !(unsigned __int8)sub_15E0300(i) )
    {
      v26 = *(_DWORD *)(i + 32);
      v27 = (unsigned int)v29;
      if ( (unsigned int)v29 >= HIDWORD(v29) )
      {
        sub_16CD150((__int64)&v28, v30, 0, 4, v24, v25);
        v27 = (unsigned int)v29;
      }
      *(_DWORD *)&v28[4 * v27] = v26;
      LODWORD(v29) = v29 + 1;
    }
  }
  if ( (_DWORD)v29 )
  {
    for ( j = *(_QWORD *)(a2 + 8); j; j = *(_QWORD *)(j + 8) )
    {
      v9 = (unsigned __int64)sub_1648700(j);
      v10 = *(_BYTE *)(v9 + 16);
      if ( v10 > 0x17u )
      {
        if ( v10 == 78 )
        {
          v11 = v9 | 4;
          goto LABEL_23;
        }
        if ( v10 == 29 )
        {
          v11 = v9 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_23:
          v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v11 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          {
            v13 = (v11 & 4) == 0;
            v14 = v12 - 24;
            if ( v13 )
              v14 = v12 - 72;
            if ( v14 == j && (_DWORD)v29 )
            {
              v15 = 0;
              v16 = 4LL * (unsigned int)v29;
              do
              {
                v22 = 24LL * *(unsigned int *)&v28[v15];
                v23 = sub_1599EF0(**(__int64 ****)(v12 + v22 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
                if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
                  v17 = *(_QWORD *)(v12 - 8);
                else
                  v17 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
                v18 = (__int64 *)(v22 + v17);
                if ( *v18 )
                {
                  v19 = v18[1];
                  v20 = v18[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v20 = v19;
                  if ( v19 )
                    *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
                }
                *v18 = v23;
                if ( v23 )
                {
                  v21 = *(_QWORD *)(v23 + 8);
                  v18[1] = v21;
                  if ( v21 )
                    *(_QWORD *)(v21 + 16) = (unsigned __int64)(v18 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
                  v18[2] = (v23 + 8) | v18[2] & 3;
                  *(_QWORD *)(v23 + 8) = v18;
                }
                v15 += 4;
              }
              while ( v16 != v15 );
              v2 = 1;
            }
          }
        }
      }
    }
  }
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  return v2;
}
