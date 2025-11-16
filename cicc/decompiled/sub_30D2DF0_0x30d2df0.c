// Function: sub_30D2DF0
// Address: 0x30d2df0
//
__int64 __fastcall sub_30D2DF0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 **v4; // r10
  __int64 v5; // r14
  __int64 v6; // r8
  __int64 v7; // r14
  __int64 v8; // r14
  __int64 v9; // r11
  __int64 v10; // r13
  unsigned __int8 **v11; // rcx
  int v12; // eax
  unsigned __int8 **v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // r14
  int v16; // edx
  int v17; // r13d
  __int64 v18; // rax
  __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rsi
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r9
  int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // rdi
  __int64 result; // rax
  int v33; // edx
  int v34; // r10d
  int v35; // r8d
  __int64 v36; // [rsp+0h] [rbp-80h]
  __int64 **v37; // [rsp+8h] [rbp-78h]
  unsigned __int8 **v38; // [rsp+20h] [rbp-60h] BYREF
  __int64 v39; // [rsp+28h] [rbp-58h]
  _BYTE v40[80]; // [rsp+30h] [rbp-50h] BYREF

  v3 = (__int64 *)a2;
  v4 = *(__int64 ***)(a1 + 8);
  v5 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v6 = *(_QWORD *)(a2 - 8);
    v7 = v6 + v5;
  }
  else
  {
    v6 = a2 - v5;
    v7 = a2;
  }
  v8 = v7 - v6;
  v38 = (unsigned __int8 **)v40;
  v9 = v8 >> 5;
  v39 = 0x400000000LL;
  v10 = v8 >> 5;
  if ( (unsigned __int64)v8 > 0x80 )
  {
    v36 = v6;
    v37 = v4;
    sub_C8D5F0((__int64)&v38, v40, v8 >> 5, 8u, v6, (__int64)v40);
    v13 = v38;
    v12 = v39;
    v9 = v8 >> 5;
    v4 = v37;
    v6 = v36;
    v11 = &v38[(unsigned int)v39];
  }
  else
  {
    v11 = (unsigned __int8 **)v40;
    v12 = 0;
    v13 = (unsigned __int8 **)v40;
  }
  if ( v8 > 0 )
  {
    v14 = 0;
    do
    {
      v11[v14 / 8] = *(unsigned __int8 **)(v6 + 4 * v14);
      v14 += 8LL;
      --v10;
    }
    while ( v10 );
    v13 = v38;
    v12 = v39;
  }
  LODWORD(v39) = v12 + v9;
  v15 = sub_DFCEF0(v4, (unsigned __int8 *)a2, v13, (unsigned int)(v12 + v9), 3);
  v17 = v16;
  if ( v38 != (unsigned __int8 **)v40 )
    _libc_free((unsigned __int64)v38);
  if ( v17 || (result = 1, v15) )
  {
    v18 = 4LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    v19 = (__int64 *)(a2 - v18 * 8);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v19 = *(__int64 **)(a2 - 8);
      v3 = &v19[v18];
    }
    for ( ; v3 != v19; v19 += 4 )
    {
      v20 = *(unsigned int *)(a1 + 192);
      v21 = *v19;
      v22 = *(_QWORD *)(a1 + 176);
      if ( (_DWORD)v20 )
      {
        v23 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v21 == *v24 )
        {
LABEL_17:
          if ( v24 != (__int64 *)(v22 + 16 * v20) )
          {
            v26 = *(_DWORD *)(a1 + 224);
            v27 = v24[1];
            v28 = *(_QWORD *)(a1 + 208);
            if ( v26 )
            {
              v29 = v26 - 1;
              v30 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
              v31 = *(_QWORD *)(v28 + 8LL * v30);
              if ( v27 == v31 )
              {
LABEL_20:
                if ( v27 )
                  sub_30D1890(a1, v27);
              }
              else
              {
                v35 = 1;
                while ( v31 != -4096 )
                {
                  v30 = v29 & (v35 + v30);
                  v31 = *(_QWORD *)(v28 + 8LL * v30);
                  if ( v27 == v31 )
                    goto LABEL_20;
                  ++v35;
                }
              }
            }
          }
        }
        else
        {
          v33 = 1;
          while ( v25 != -4096 )
          {
            v34 = v33 + 1;
            v23 = (v20 - 1) & (v33 + v23);
            v24 = (__int64 *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( v21 == *v24 )
              goto LABEL_17;
            v33 = v34;
          }
        }
      }
    }
    return 0;
  }
  return result;
}
