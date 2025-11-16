// Function: sub_1DCCA50
// Address: 0x1dcca50
//
unsigned __int64 __fastcall sub_1DCCA50(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  __int64 *v8; // r13
  unsigned __int64 result; // rax
  __int64 *v10; // rdi
  __int64 *v11; // rcx
  __int64 v12; // rax
  unsigned int v13; // esi
  unsigned int v14; // edx
  _BYTE *v15; // rsi
  __int64 *v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rax
  unsigned int v19; // [rsp+8h] [rbp-48h]
  __int64 *i; // [rsp+8h] [rbp-48h]
  _QWORD v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = *(_DWORD *)(a3 + 48);
  v8 = (__int64 *)sub_1DCC790((char *)a1, a2);
  result = v8[5];
  if ( v8[4] != result && a3 == *(_QWORD *)(*(_QWORD *)(result - 8) + 24LL) )
  {
    *(_QWORD *)(result - 8) = a4;
    return result;
  }
  result = sub_1E69D00(*(_QWORD *)(a1 + 352), a2);
  if ( a3 != *(_QWORD *)(result + 24) )
  {
    v10 = (__int64 *)v8[1];
    v11 = v8 + 1;
    if ( v10 == v8 + 1 )
      goto LABEL_16;
    v12 = *v8;
    if ( v11 == (__int64 *)*v8 )
    {
      v12 = *(_QWORD *)(v12 + 8);
      *v8 = v12;
      v13 = *(_DWORD *)(v12 + 16);
      v14 = v19 >> 7;
      if ( v19 >> 7 == v13 )
      {
        if ( v11 == (__int64 *)v12 )
          goto LABEL_16;
        goto LABEL_15;
      }
    }
    else
    {
      v13 = *(_DWORD *)(v12 + 16);
      v14 = v19 >> 7;
      if ( v19 >> 7 == v13 )
      {
LABEL_15:
        if ( (*(_QWORD *)(v12 + 8LL * ((v19 >> 6) & 1) + 24) & (1LL << v19)) != 0 )
        {
LABEL_20:
          result = *(_QWORD *)(a3 + 72);
          v16 = *(__int64 **)(a3 + 64);
          for ( i = (__int64 *)result; i != v16; result = sub_1DCBEC0(a1, v8, *(_QWORD *)(v18 + 24), v17) )
          {
            v17 = *v16++;
            v18 = sub_1E69D00(*(_QWORD *)(a1 + 352), a2);
          }
          return result;
        }
LABEL_16:
        v21[0] = a4;
        v15 = (_BYTE *)v8[5];
        if ( v15 == (_BYTE *)v8[6] )
        {
          sub_1DCC370((__int64)(v8 + 4), v15, v21);
        }
        else
        {
          if ( v15 )
          {
            *(_QWORD *)v15 = a4;
            v15 = (_BYTE *)v8[5];
          }
          v8[5] = (__int64)(v15 + 8);
        }
        goto LABEL_20;
      }
    }
    if ( v14 < v13 )
    {
      if ( v10 == (__int64 *)v12 )
      {
        *v8 = v12;
        goto LABEL_14;
      }
      do
        v12 = *(_QWORD *)(v12 + 8);
      while ( v10 != (__int64 *)v12 && *(_DWORD *)(v12 + 16) > v14 );
    }
    else
    {
      if ( v11 == (__int64 *)v12 )
      {
LABEL_31:
        *v8 = v12;
        goto LABEL_16;
      }
      while ( v14 > v13 )
      {
        v12 = *(_QWORD *)v12;
        if ( v11 == (__int64 *)v12 )
          goto LABEL_31;
        v13 = *(_DWORD *)(v12 + 16);
      }
    }
    *v8 = v12;
    if ( v11 == (__int64 *)v12 )
      goto LABEL_16;
LABEL_14:
    if ( *(_DWORD *)(v12 + 16) != v14 )
      goto LABEL_16;
    goto LABEL_15;
  }
  return result;
}
