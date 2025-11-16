// Function: sub_27CF2B0
// Address: 0x27cf2b0
//
__int64 __fastcall sub_27CF2B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  unsigned int v8; // r8d
  __int64 v9; // rdx
  __int64 *v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  bool v17; // zf
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 result; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // [rsp+10h] [rbp-20h] BYREF
  __int64 v32; // [rsp+18h] [rbp-18h]

  v5 = *(_QWORD *)(a2 - 32);
  if ( !v5 )
    BUG();
  if ( *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80) )
    goto LABEL_51;
  v8 = *(_DWORD *)(v5 + 36);
  v9 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL);
  v10 = *(__int64 **)(v9 + 40);
  if ( v8 > 0xE6 )
  {
    if ( v8 == 299 )
      return 0;
    if ( v8 <= 0x12B )
    {
      if ( v8 == 282 )
        goto LABEL_8;
      if ( v8 == 286 )
      {
LABEL_24:
        v31 = *(_QWORD *)(a4 + 8);
        if ( !*(_BYTE *)v5 )
        {
          v12 = 1;
LABEL_9:
          v13 = sub_B6E160(v10, v8, (__int64)&v31, v12);
          v14 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
          if ( *(_QWORD *)v14 )
          {
            v15 = *(_QWORD *)(v14 + 8);
            **(_QWORD **)(v14 + 16) = v15;
            if ( v15 )
              *(_QWORD *)(v15 + 16) = *(_QWORD *)(v14 + 16);
          }
          *(_QWORD *)v14 = a4;
          v16 = *(_QWORD *)(a4 + 16);
          *(_QWORD *)(v14 + 8) = v16;
          if ( v16 )
            *(_QWORD *)(v16 + 16) = v14 + 8;
          *(_QWORD *)(v14 + 16) = a4 + 16;
          *(_QWORD *)(a4 + 16) = v14;
          v17 = *(_QWORD *)(a2 - 32) == 0;
          *(_QWORD *)(a2 + 80) = *(_QWORD *)(v13 + 24);
          if ( !v17 )
          {
            v18 = *(_QWORD *)(a2 - 24);
            **(_QWORD **)(a2 - 16) = v18;
            if ( v18 )
              *(_QWORD *)(v18 + 16) = *(_QWORD *)(a2 - 16);
          }
          *(_QWORD *)(a2 - 32) = v13;
          v19 = *(_QWORD *)(v13 + 16);
          *(_QWORD *)(a2 - 24) = v19;
          if ( v19 )
            *(_QWORD *)(v19 + 16) = a2 - 24;
          *(_QWORD *)(a2 - 16) = v13 + 16;
          result = 1;
          *(_QWORD *)(v13 + 16) = a2 - 32;
          return result;
        }
LABEL_51:
        BUG();
      }
    }
    else
    {
      result = 0;
      if ( v8 == 8170 )
        return result;
    }
LABEL_29:
    v21 = sub_DF9C00(*(_QWORD *)(a1 + 24));
    if ( v21 )
    {
      result = 1;
      if ( a2 != v21 )
      {
        sub_BD84D0(a2, v21);
        return 1;
      }
      return result;
    }
    return 0;
  }
  if ( v8 > 0xE4 )
  {
    v30 = *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL);
    v32 = *(_QWORD *)(a4 + 8);
    v31 = v30;
    if ( *(_BYTE *)v5 )
      goto LABEL_51;
    v24 = sub_B6E160(v10, v8, (__int64)&v31, 2);
    v25 = a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( !*(_QWORD *)v25 )
      goto LABEL_38;
    goto LABEL_36;
  }
  if ( v8 != 227 )
  {
    switch ( v8 )
    {
      case 0xE4u:
LABEL_8:
        v11 = *(_QWORD *)(a2 + 8);
        v12 = 2;
        v32 = *(_QWORD *)(a4 + 8);
        v31 = v11;
        goto LABEL_9;
      case 0xABu:
        sub_BD2ED0(a2, a3, a4);
        return 1;
      case 0xCEu:
        goto LABEL_24;
    }
    goto LABEL_29;
  }
  v22 = *(_QWORD *)(a4 + 8);
  v23 = *(__int64 **)(v9 + 40);
  v31 = *(_QWORD *)(a2 + 8);
  v32 = v22;
  v24 = sub_B6E160(v23, 0xE3u, (__int64)&v31, 2);
  v25 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)v25 )
  {
LABEL_36:
    v26 = *(_QWORD *)(v25 + 8);
    **(_QWORD **)(v25 + 16) = v26;
    if ( v26 )
      *(_QWORD *)(v26 + 16) = *(_QWORD *)(v25 + 16);
  }
LABEL_38:
  *(_QWORD *)v25 = a4;
  v27 = *(_QWORD *)(a4 + 16);
  *(_QWORD *)(v25 + 8) = v27;
  if ( v27 )
    *(_QWORD *)(v27 + 16) = v25 + 8;
  *(_QWORD *)(v25 + 16) = a4 + 16;
  *(_QWORD *)(a4 + 16) = v25;
  v17 = *(_QWORD *)(a2 - 32) == 0;
  *(_QWORD *)(a2 + 80) = *(_QWORD *)(v24 + 24);
  if ( !v17 )
  {
    v28 = *(_QWORD *)(a2 - 24);
    **(_QWORD **)(a2 - 16) = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(a2 - 16);
  }
  *(_QWORD *)(a2 - 32) = v24;
  v29 = *(_QWORD *)(v24 + 16);
  *(_QWORD *)(a2 - 24) = v29;
  if ( v29 )
    *(_QWORD *)(v29 + 16) = a2 - 24;
  *(_QWORD *)(a2 - 16) = v24 + 16;
  *(_QWORD *)(v24 + 16) = a2 - 32;
  return 1;
}
