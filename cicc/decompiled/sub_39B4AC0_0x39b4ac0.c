// Function: sub_39B4AC0
// Address: 0x39b4ac0
//
__int64 __fastcall sub_39B4AC0(
        __int64 *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 *a8,
        __int64 a9)
{
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  char v16; // di
  int v17; // edx
  unsigned __int64 v18; // rsi
  char v19; // al
  int v20; // r8d
  __int64 result; // rax
  __int64 v22; // r15
  int v23; // esi
  int v24; // ebx
  int v25; // eax
  __int64 v26; // rcx
  int v27; // r15d
  int v28; // r14d
  __int64 v29; // rdx
  int v30; // r14d
  int v31; // r15d
  __int64 v32; // rdx
  __int64 v33; // [rsp+0h] [rbp-50h]
  int v34; // [rsp+Ch] [rbp-44h]
  int v35; // [rsp+18h] [rbp-38h]

  v12 = a1[2];
  v13 = (unsigned int)sub_1F43D70(v12, a2);
  v15 = sub_1F43D80(v12, *a1, a3, v14);
  v16 = *(_BYTE *)(a3 + 8);
  v17 = v15;
  v18 = HIDWORD(v15);
  v19 = v16;
  if ( v16 == 16 )
    v19 = *(_BYTE *)(**(_QWORD **)(a3 + 16) + 8LL);
  v20 = 1;
  result = (unsigned int)((unsigned __int8)(v19 - 1) < 6u) + 1;
  if ( (_BYTE)v18 == 1
    || (_BYTE)v18 && (v20 = (unsigned __int8)v18, *(_QWORD *)(v12 + 8LL * (unsigned __int8)v18 + 120)) )
  {
    if ( (unsigned int)v13 > 0x102 )
    {
      if ( *(_QWORD *)(v12 + 8LL * v20 + 120) )
        return (unsigned int)(2 * v17 * result);
    }
    else
    {
      v22 = v12 + 259LL * (unsigned int)v20 + v13;
      if ( *(_BYTE *)(v22 + 2422) <= 1u )
        return (unsigned int)(v17 * result);
      if ( *(_QWORD *)(v12 + 8LL * v20 + 120) && *(_BYTE *)(v22 + 2422) != 2 )
        return (unsigned int)(2 * v17 * result);
    }
  }
  if ( v16 == 16 )
  {
    v23 = a2;
    v33 = *(_QWORD *)(a3 + 32);
    v24 = 0;
    v25 = sub_39B4AC0((_DWORD)a1, v23, **(_QWORD **)(a3 + 16), 0, 0, 0, 0, 0, 0);
    v26 = *(_QWORD *)(a3 + 32);
    v34 = v25;
    v27 = v26;
    if ( (int)v26 <= 0 )
    {
      if ( !a9 )
        return (unsigned int)(v24 + v33 * v34);
    }
    else
    {
      v28 = 0;
      do
      {
        v29 = a3;
        if ( *(_BYTE *)(a3 + 8) == 16 )
          v29 = **(_QWORD **)(a3 + 16);
        ++v28;
        v24 += sub_1F43D80(a1[2], *a1, v29, v26);
      }
      while ( v27 != v28 );
      v26 = *(_QWORD *)(a3 + 32);
      if ( !a9 )
      {
        v35 = *(_QWORD *)(a3 + 32);
        if ( (int)v26 > 0 )
        {
          v30 = 0;
          v31 = 0;
          do
          {
            v32 = a3;
            if ( *(_BYTE *)(a3 + 8) == 16 )
              v32 = **(_QWORD **)(a3 + 16);
            ++v31;
            v30 += sub_1F43D80(a1[2], *a1, v32, v26);
          }
          while ( v35 != v31 );
          v24 += v30;
        }
        return (unsigned int)(v24 + v33 * v34);
      }
    }
    v24 += sub_39B48D0(a1, a8, a9, v26);
    return (unsigned int)(v24 + v33 * v34);
  }
  return result;
}
