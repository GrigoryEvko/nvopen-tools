// Function: sub_3702790
// Address: 0x3702790
//
unsigned __int64 *__fastcall sub_3702790(
        unsigned __int64 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rsi
  unsigned int v10; // esi
  const __m128i *v11; // r8
  unsigned __int64 v12; // rax
  unsigned int v13; // ebx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned int v18; // ebx
  unsigned __int64 v20[5]; // [rsp+18h] [rbp-28h] BYREF

  v7 = a2[7] == 0;
  v8 = a2[5];
  if ( !v7 )
  {
    if ( !v8 && !a2[6] )
    {
      v10 = *(_DWORD *)(a3 + 8);
      v11 = (const __m128i *)a4;
      v12 = *(_QWORD *)a3;
      if ( *(_BYTE *)(a3 + 12) )
      {
        if ( v10 > 0x40 )
          v12 = *(_QWORD *)v12;
        v20[0] = v12;
        sub_3701BE0(a2, v20, (const __m128i *)a4, a4, a4, a6);
      }
      else
      {
        if ( v10 > 0x40 )
        {
          v16 = *(_QWORD *)v12;
        }
        else
        {
          v16 = 0;
          if ( v10 )
          {
            a4 = 64 - v10;
            v16 = (__int64)(v12 << (64 - (unsigned __int8)v10)) >> (64 - (unsigned __int8)v10);
          }
        }
        v20[0] = v16;
        sub_3701890(a2, v20, v11, a4, (__int64)v11, a6);
      }
      *a1 = 1;
      return a1;
    }
LABEL_3:
    sub_3708A50(a1, v8, a3);
    return a1;
  }
  if ( !a2[6] || v8 )
    goto LABEL_3;
  v13 = *(_DWORD *)(a3 + 8);
  if ( *(_BYTE *)(a3 + 12) )
  {
    if ( v13 > 0x40 )
    {
      v18 = v13 - sub_C444A0(a3);
      v17 = -1;
      if ( v18 <= 0x40 )
        v17 = **(_QWORD **)a3;
    }
    else
    {
      v17 = *(_QWORD *)a3;
    }
    v20[0] = v17;
    sub_3702300(a1, (__int64)a2, v20);
  }
  else
  {
    v14 = 0x8000000000000000LL;
    if ( v13 <= 0x40 )
    {
      v15 = *(_QWORD *)a3;
      v14 = 0;
      if ( v13 )
        v14 = (__int64)(v15 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13);
    }
    v20[0] = v14;
    sub_3701ED0(a1, (__int64)a2, v20);
  }
  return a1;
}
