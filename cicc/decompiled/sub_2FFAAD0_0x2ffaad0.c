// Function: sub_2FFAAD0
// Address: 0x2ffaad0
//
__int64 __fastcall sub_2FFAAD0(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rax
  _BYTE *v9; // rsi
  __int64 v10; // r8
  __int64 result; // rax
  _BYTE *v12; // rbx
  _BYTE *v13; // r12
  _BYTE *v14; // r14
  _BYTE *v15; // rbx
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v8 = sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v16[0] = a3;
  v9 = sub_2FF8C10(*(_QWORD **)(v8 + 32), *(_QWORD *)(v8 + 40), v16);
  result = 0;
  if ( *(_BYTE **)(v10 + 40) != v9 )
  {
    sub_2E25970(v10 + 32, v9);
    v12 = *(_BYTE **)(a3 + 32);
    v13 = &v12[40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF)];
    if ( v12 != v13 )
    {
      while ( 1 )
      {
        v14 = v12;
        if ( sub_2DADC00(v12) )
          break;
        v12 += 40;
        if ( v13 == v12 )
          return 1;
      }
      while ( v13 != v14 )
      {
        if ( a2 == *((_DWORD *)v14 + 2) )
        {
          v14[3] &= ~0x40u;
          return 1;
        }
        v15 = v14 + 40;
        if ( v14 + 40 == v13 )
          return 1;
        while ( 1 )
        {
          v14 = v15;
          if ( sub_2DADC00(v15) )
            break;
          v15 += 40;
          if ( v13 == v15 )
            return 1;
        }
      }
    }
    return 1;
  }
  return result;
}
