// Function: sub_25177C0
// Address: 0x25177c0
//
__int64 __fastcall sub_25177C0(__int64 a1, char a2)
{
  __int64 v2; // rbx
  __int64 v3; // rsi
  __int64 v4; // rdi
  int v5; // r10d
  __int64 *v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // r12
  int v12; // eax
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 *v14; // [rsp+8h] [rbp-68h] BYREF
  __int64 v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h]
  unsigned int v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  __int64 *v20; // [rsp+38h] [rbp-38h]
  __int64 v21; // [rsp+40h] [rbp-30h]
  int v22; // [rsp+48h] [rbp-28h]
  char v23; // [rsp+4Ch] [rbp-24h]
  __int64 v24; // [rsp+50h] [rbp-20h] BYREF

  v2 = a1;
  if ( !(_BYTE)qword_4FEEC88 && !a2 || !sub_250E810(a1) )
    return 0;
  v23 = 1;
  v20 = &v24;
  v21 = 0x100000002LL;
  v22 = 0;
  v24 = a1;
  v19 = 1;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_25167D0((__int64)&v19, (__int64)&v15);
  v3 = v18;
  v13 = a1;
  if ( !v18 )
  {
    ++v15;
    v14 = 0;
    goto LABEL_25;
  }
  v4 = v16;
  v5 = 1;
  v6 = 0;
  v7 = (v18 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v8 = (__int64 *)(v16 + 16LL * v7);
  v9 = *v8;
  if ( v2 != *v8 )
  {
    while ( v9 != -4096 )
    {
      if ( !v6 && v9 == -8192 )
        v6 = v8;
      v7 = (v18 - 1) & (v5 + v7);
      v8 = (__int64 *)(v16 + 16LL * v7);
      v9 = *v8;
      if ( v2 == *v8 )
        goto LABEL_6;
      ++v5;
    }
    if ( !v6 )
      v6 = v8;
    ++v15;
    v12 = v17 + 1;
    v14 = v6;
    if ( 4 * ((int)v17 + 1) < 3 * v18 )
    {
      if ( v18 - HIDWORD(v17) - v12 > v18 >> 3 )
      {
LABEL_21:
        LODWORD(v17) = v12;
        if ( *v6 != -4096 )
          --HIDWORD(v17);
        *v6 = v2;
        v10 = 0;
        v6[1] = 0;
        v3 = v18;
        v4 = v16;
        goto LABEL_7;
      }
LABEL_26:
      sub_9E0010((__int64)&v15, v3);
      sub_25108F0((__int64)&v15, &v13, &v14);
      v2 = v13;
      v6 = v14;
      v12 = v17 + 1;
      goto LABEL_21;
    }
LABEL_25:
    LODWORD(v3) = 2 * v18;
    goto LABEL_26;
  }
LABEL_6:
  v10 = v8[1];
LABEL_7:
  sub_C7D6A0(v4, 16 * v3, 8);
  if ( !v23 )
    _libc_free((unsigned __int64)v20);
  return v10;
}
