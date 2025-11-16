// Function: sub_198E380
// Address: 0x198e380
//
__int64 __fastcall sub_198E380(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char *v5; // r12
  char *v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // r14
  int v9; // edx
  _QWORD *v10; // r15
  char *v11; // r14
  __int64 v12; // rax
  _QWORD *v13; // r12
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // r13
  int v17; // ecx
  int v18; // ecx
  __int64 v19; // rdi
  unsigned int v20; // esi
  __int64 *v21; // rax
  __int64 v22; // r11
  _QWORD *v23; // rbx
  __int64 v24; // rax
  int v26; // eax
  int v27; // edx
  unsigned __int8 v30; // [rsp+20h] [rbp-1D0h]
  char *v31; // [rsp+20h] [rbp-1D0h]
  _BYTE *v33; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v34; // [rsp+38h] [rbp-1B8h]
  _BYTE v35[432]; // [rsp+40h] [rbp-1B0h] BYREF

  v5 = *(char **)(a1 + 40);
  v6 = *(char **)(a1 + 32);
  v34 = 0x1000000000LL;
  v33 = v35;
  v7 = (v5 - v6) >> 3;
  if ( (unsigned __int64)(v5 - v6) > 0x80 )
  {
    v31 = v6;
    sub_170B450((__int64)&v33, (v5 - v6) >> 3);
    v9 = v34;
    v8 = v33;
    v6 = v31;
    v10 = &v33[24 * (unsigned int)v34];
  }
  else
  {
    v8 = v35;
    v9 = 0;
    v10 = v35;
  }
  if ( v6 != v5 )
  {
    v11 = v6;
    do
    {
      if ( v10 )
      {
        v12 = *(_QWORD *)v11;
        *v10 = 6;
        v10[1] = 0;
        v10[2] = v12;
        if ( v12 != 0 && v12 != -8 && v12 != -16 )
          sub_164C220((__int64)v10);
      }
      v11 += 8;
      v10 += 3;
    }
    while ( v5 != v11 );
    v8 = v33;
    v9 = v34;
  }
  v30 = 0;
  LODWORD(v34) = v7 + v9;
  v13 = &v8[3 * (unsigned int)(v7 + v9)];
  if ( v13 != v8 )
  {
    do
    {
      v14 = v8[2];
      if ( v14 )
      {
        v15 = sub_157F0B0(v8[2]);
        v16 = v15;
        if ( v15 )
        {
          if ( sub_157F1C0(v15) )
          {
            v17 = *(_DWORD *)(a3 + 24);
            if ( v17 )
            {
              v18 = v17 - 1;
              v19 = *(_QWORD *)(a3 + 8);
              v20 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v21 = (__int64 *)(v19 + 16LL * v20);
              v22 = *v21;
              if ( v16 == *v21 )
              {
LABEL_17:
                if ( a1 == v21[1] )
                {
                  sub_1AA7EA0(v14, a2, a3, 0, 0);
                  sub_1465150(a4, a1);
                  v30 = 1;
                }
              }
              else
              {
                v26 = 1;
                while ( v22 != -8 )
                {
                  v27 = v26 + 1;
                  v20 = v18 & (v26 + v20);
                  v21 = (__int64 *)(v19 + 16LL * v20);
                  v22 = *v21;
                  if ( v16 == *v21 )
                    goto LABEL_17;
                  v26 = v27;
                }
              }
            }
          }
        }
      }
      v8 += 3;
    }
    while ( v8 != v13 );
    v23 = v33;
    v8 = &v33[24 * (unsigned int)v34];
    if ( v33 != (_BYTE *)v8 )
    {
      do
      {
        v24 = *(v8 - 1);
        v8 -= 3;
        if ( v24 != 0 && v24 != -8 && v24 != -16 )
          sub_1649B30(v8);
      }
      while ( v23 != v8 );
      v8 = v33;
    }
  }
  if ( v8 != (_QWORD *)v35 )
    _libc_free((unsigned __int64)v8);
  return v30;
}
