// Function: sub_34E1D10
// Address: 0x34e1d10
//
_DWORD *__fastcall sub_34E1D10(__int64 a1, _DWORD *a2, __int64 a3, const __m128i *a4)
{
  unsigned __int8 *v5; // r15
  __int64 v6; // rax
  _DWORD *v7; // rcx
  _DWORD *result; // rax
  __int64 v9; // r14
  _DWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int8 v13; // al
  unsigned __int8 **v14; // rdx
  unsigned __int8 *v15; // rdx
  __int64 v16; // rax
  _DWORD *v17; // [rsp+0h] [rbp-80h]
  int v18; // [rsp+Ch] [rbp-74h]
  _DWORD *v19; // [rsp+10h] [rbp-70h]
  __m128i v22; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+40h] [rbp-40h]

  v5 = (unsigned __int8 *)a4[1].m128i_i64[0];
  v18 = *a2;
  v6 = *(_QWORD *)(a1 + 152);
  v7 = *(_DWORD **)(v6 + 328);
  result = (_DWORD *)(v6 + 320);
  v17 = result;
  v19 = v7;
  if ( v7 != result )
  {
    do
    {
      v9 = *((_QWORD *)v19 + 7);
      v10 = v19 + 12;
      if ( (_DWORD *)v9 != v19 + 12 )
      {
        while ( 1 )
        {
          if ( (unsigned __int16)(*(_WORD *)(v9 + 68) - 14) > 4u )
          {
            v11 = sub_B10CD0(v9 + 56);
            v12 = v11;
            if ( v11 )
            {
              v13 = *(_BYTE *)(v11 - 16);
              v14 = (v13 & 2) != 0
                  ? *(unsigned __int8 ***)(v12 - 32)
                  : (unsigned __int8 **)(v12 - 16 - 8LL * ((v13 >> 2) & 0xF));
              v15 = *v14;
              v16 = a4[1].m128i_i64[0];
              v22 = _mm_loadu_si128(a4);
              v23 = v16;
              if ( (unsigned __int8)sub_3143CD0(a1, v12, v15, v5, a3, v22.m128i_i64, a2) )
                break;
            }
          }
          if ( (*(_BYTE *)v9 & 4) != 0 )
          {
            v9 = *(_QWORD *)(v9 + 8);
            if ( v10 == (_DWORD *)v9 )
              break;
          }
          else
          {
            while ( (*(_BYTE *)(v9 + 44) & 8) != 0 )
              v9 = *(_QWORD *)(v9 + 8);
            v9 = *(_QWORD *)(v9 + 8);
            if ( v10 == (_DWORD *)v9 )
              break;
          }
        }
      }
      result = a2;
      if ( *a2 != v18 )
        break;
      result = (_DWORD *)*((_QWORD *)v19 + 1);
      v19 = result;
    }
    while ( v17 != result );
  }
  return result;
}
