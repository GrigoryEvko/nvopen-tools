// Function: sub_2E29FE0
// Address: 0x2e29fe0
//
void __fastcall sub_2E29FE0(__m128i *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v9; // r12
  __int64 v10; // rax
  __int64 *v11; // rsi
  __int64 *v12; // rax
  unsigned int v13; // ecx
  unsigned int v14; // edx
  _BYTE *v15; // rsi
  __int64 *v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // [rsp+8h] [rbp-48h]
  __int64 *v20; // [rsp+8h] [rbp-48h]
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = *(_DWORD *)(a3 + 24);
  v9 = (__int64 *)sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v10 = v9[5];
  if ( v9[4] != v10 && a3 == *(_QWORD *)(*(_QWORD *)(v10 - 8) + 24LL) )
  {
    *(_QWORD *)(v10 - 8) = a4;
    return;
  }
  if ( a3 != *(_QWORD *)(sub_2EBEE10(a1[5].m128i_i64[1], a2) + 24) )
  {
    v11 = (__int64 *)*v9;
    if ( v9 == (__int64 *)*v9 )
      goto LABEL_16;
    v12 = (__int64 *)v9[3];
    if ( v9 == v12 )
    {
      v12 = (__int64 *)v9[1];
      v9[3] = (__int64)v12;
      v13 = *((_DWORD *)v12 + 4);
      v14 = v19 >> 7;
      if ( v19 >> 7 == v13 )
      {
        if ( v9 == v12 )
          goto LABEL_16;
        goto LABEL_15;
      }
    }
    else
    {
      v13 = *((_DWORD *)v12 + 4);
      v14 = v19 >> 7;
      if ( v19 >> 7 == v13 )
        goto LABEL_15;
    }
    if ( v13 > v14 )
    {
      if ( v11 != v12 )
      {
        while ( 1 )
        {
          v12 = (__int64 *)v12[1];
          if ( v11 == v12 )
            break;
          if ( *((_DWORD *)v12 + 4) <= v14 )
            goto LABEL_13;
        }
      }
      v9[3] = (__int64)v12;
    }
    else
    {
      if ( v9 == v12 )
      {
LABEL_32:
        v9[3] = (__int64)v12;
        goto LABEL_16;
      }
      while ( v13 < v14 )
      {
        v12 = (__int64 *)*v12;
        if ( v9 == v12 )
          goto LABEL_32;
        v13 = *((_DWORD *)v12 + 4);
      }
LABEL_13:
      v9[3] = (__int64)v12;
      if ( v9 == v12 )
        goto LABEL_16;
    }
    if ( *((_DWORD *)v12 + 4) != v14 )
      goto LABEL_16;
LABEL_15:
    if ( (v12[((v19 >> 6) & 1) + 3] & (1LL << v19)) != 0 )
    {
LABEL_20:
      v16 = *(__int64 **)(a3 + 64);
      v20 = &v16[*(unsigned int *)(a3 + 72)];
      while ( v20 != v16 )
      {
        v17 = *v16++;
        v18 = sub_2EBEE10(a1[5].m128i_i64[1], a2);
        sub_2E25B90((__int64)a1, v9, *(_QWORD *)(v18 + 24), v17);
      }
      return;
    }
LABEL_16:
    v21[0] = a4;
    v15 = (_BYTE *)v9[5];
    if ( v15 == (_BYTE *)v9[6] )
    {
      sub_2E26050((__int64)(v9 + 4), v15, v21);
    }
    else
    {
      if ( v15 )
      {
        *(_QWORD *)v15 = a4;
        v15 = (_BYTE *)v9[5];
      }
      v9[5] = (__int64)(v15 + 8);
    }
    goto LABEL_20;
  }
}
