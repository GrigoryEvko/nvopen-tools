// Function: sub_2F32200
// Address: 0x2f32200
//
__int64 __fastcall sub_2F32200(__m128i *a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // r8
  _BYTE *v10; // rbx
  _BYTE *v11; // r12
  _BYTE *v12; // r14
  _BYTE *v13; // rbx
  __int64 v14[5]; // [rsp+8h] [rbp-28h] BYREF

  v7 = sub_2E29D60(a1, a2, a3, a4, a5, a6);
  v14[0] = a3;
  result = (__int64)sub_2F31180(*(_QWORD **)(v7 + 32), *(_QWORD *)(v7 + 40), v14);
  if ( *(_QWORD *)(v9 + 40) != result )
  {
    sub_2E25970(v9 + 32, (_BYTE *)result);
    v10 = *(_BYTE **)(a3 + 32);
    result = 5LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF);
    v11 = &v10[40 * (*(_DWORD *)(a3 + 40) & 0xFFFFFF)];
    if ( v10 != v11 )
    {
      while ( 1 )
      {
        v12 = v10;
        result = sub_2DADC00(v10);
        if ( (_BYTE)result )
          break;
        v10 += 40;
        if ( v11 == v10 )
          return result;
      }
      while ( v11 != v12 )
      {
        if ( *((_DWORD *)v12 + 2) == a2 )
        {
          v12[3] &= ~0x40u;
          return result;
        }
        v13 = v12 + 40;
        if ( v12 + 40 == v11 )
          return result;
        while ( 1 )
        {
          v12 = v13;
          result = sub_2DADC00(v13);
          if ( (_BYTE)result )
            break;
          v13 += 40;
          if ( v11 == v13 )
            return result;
        }
      }
    }
  }
  return result;
}
