// Function: sub_216EA10
// Address: 0x216ea10
//
__int64 __fastcall sub_216EA10(__int64 *a1, unsigned int a2, __int64 a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rcx
  unsigned __int64 v11; // rax
  char v12; // di
  int v13; // edx
  unsigned __int64 v14; // rsi
  char v15; // al
  int v16; // r8d
  __int64 result; // rax
  __int64 v18; // r15
  __int64 v19; // rsi
  int v20; // ebx
  int v21; // eax
  __int64 v22; // rcx
  int v23; // r15d
  int v24; // r14d
  __int64 v25; // rdx
  int v26; // r14d
  int v27; // r15d
  __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-50h]
  int v30; // [rsp+Ch] [rbp-44h]
  int v33; // [rsp+18h] [rbp-38h]

  v8 = a1[2];
  v9 = (unsigned int)sub_1F43D70(v8, a2);
  v11 = sub_1F43D80(v8, *a1, a3, v10);
  v12 = *(_BYTE *)(a3 + 8);
  v13 = v11;
  v14 = HIDWORD(v11);
  v15 = v12;
  if ( v12 == 16 )
    v15 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  v16 = 1;
  result = (unsigned int)((unsigned __int8)(v15 - 1) < 6u) + 1;
  if ( (_BYTE)v14 == 1 || (_BYTE)v14 && (v16 = (unsigned __int8)v14, *(_QWORD *)(v8 + 8LL * (unsigned __int8)v14 + 120)) )
  {
    if ( (unsigned int)v9 > 0x102 )
    {
      if ( *(_QWORD *)(v8 + 8LL * v16 + 120) )
        return (unsigned int)(2 * v13 * result);
    }
    else
    {
      v18 = v8 + 259LL * (unsigned int)v16 + v9;
      if ( *(_BYTE *)(v18 + 2422) <= 1u )
        return (unsigned int)(v13 * result);
      if ( *(_QWORD *)(v8 + 8LL * v16 + 120) && *(_BYTE *)(v18 + 2422) != 2 )
        return (unsigned int)(2 * v13 * result);
    }
  }
  if ( v12 == 16 )
  {
    v19 = a2;
    v29 = *(_QWORD *)(a3 + 32);
    v20 = 0;
    v21 = sub_216EC30(a1, v19, **(_QWORD **)(a3 + 16), 0, 0, 0, 0, 0, 0);
    v22 = *(_QWORD *)(a3 + 32);
    v30 = v21;
    v23 = v22;
    if ( (int)v22 <= 0 )
    {
      if ( !a5 )
        return (unsigned int)(v20 + v29 * v30);
    }
    else
    {
      v24 = 0;
      do
      {
        v25 = a3;
        if ( *(_BYTE *)(a3 + 8) == 16 )
          v25 = **(_QWORD **)(a3 + 16);
        ++v24;
        v20 += sub_1F43D80(a1[2], *a1, v25, v22);
      }
      while ( v23 != v24 );
      v22 = *(_QWORD *)(a3 + 32);
      if ( !a5 )
      {
        v33 = *(_QWORD *)(a3 + 32);
        if ( (int)v22 > 0 )
        {
          v26 = 0;
          v27 = 0;
          do
          {
            v28 = a3;
            if ( *(_BYTE *)(a3 + 8) == 16 )
              v28 = **(_QWORD **)(a3 + 16);
            ++v27;
            v26 += sub_1F43D80(a1[2], *a1, v28, v22);
          }
          while ( v33 != v27 );
          v20 += v26;
        }
        return (unsigned int)(v20 + v29 * v30);
      }
    }
    v20 += sub_2169210(a1, a4, a5, v22);
    return (unsigned int)(v20 + v29 * v30);
  }
  return result;
}
