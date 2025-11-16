// Function: sub_214FBF0
// Address: 0x214fbf0
//
__int64 __fastcall sub_214FBF0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  unsigned __int64 v5; // rcx
  char *v6; // r8
  __m128i *v7; // rax
  __int64 v8; // rcx
  _QWORD *v9; // rdi
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  char v13; // [rsp+14h] [rbp-6Ch] BYREF
  _BYTE v14[11]; // [rsp+15h] [rbp-6Bh] BYREF
  _QWORD *v15; // [rsp+20h] [rbp-60h]
  __int64 v16; // [rsp+28h] [rbp-58h]
  _QWORD v17[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18[2]; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v19[6]; // [rsp+50h] [rbp-30h] BYREF

  switch ( *(_BYTE *)(a3 + 8) )
  {
    case 0:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 9:
    case 0xA:
    case 0xC:
    case 0xD:
    case 0xE:
    case 0xF:
      v11 = *(_QWORD *)(a2 + 232);
      if ( !*(_BYTE *)(v11 + 936) || *(_DWORD *)(a3 + 8) >> 8 == 3 && *(_DWORD *)(v11 + 83264) == 32 )
      {
        if ( a4 )
          sub_214B770((__int64 *)a1, "b32");
        else
          sub_214B770((__int64 *)a1, "u32");
      }
      else if ( a4 )
      {
        sub_214B770((__int64 *)a1, "b64");
      }
      else
      {
        sub_214B770((__int64 *)a1, "u64");
      }
      return a1;
    case 1:
      *(_BYTE *)(a1 + 18) = 54;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 12642;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 2:
      *(_BYTE *)(a1 + 18) = 50;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13158;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 3:
      *(_BYTE *)(a1 + 18) = 52;
      *(_QWORD *)a1 = a1 + 16;
      *(_WORD *)(a1 + 16) = 13926;
      *(_QWORD *)(a1 + 8) = 3;
      *(_BYTE *)(a1 + 19) = 0;
      return a1;
    case 0xB:
      v5 = *(_DWORD *)(a3 + 8) >> 8;
      if ( (_DWORD)v5 == 1 )
      {
        sub_214B770((__int64 *)a1, "pred");
      }
      else
      {
        v16 = 1;
        v15 = v17;
        LOWORD(v17[0]) = 117;
        if ( (_DWORD)v5 )
        {
          v6 = v14;
          do
          {
            *--v6 = v5 % 0xA + 48;
            v12 = v5;
            v5 /= 0xAu;
          }
          while ( v12 > 9 );
        }
        else
        {
          v13 = 48;
          v6 = &v13;
        }
        v18[0] = (__int64)v19;
        sub_214ADD0(v18, v6, (__int64)v14);
        v7 = (__m128i *)sub_2241130(v18, 0, 0, v15, v16);
        *(_QWORD *)a1 = a1 + 16;
        if ( (__m128i *)v7->m128i_i64[0] == &v7[1] )
        {
          *(__m128i *)(a1 + 16) = _mm_loadu_si128(v7 + 1);
        }
        else
        {
          *(_QWORD *)a1 = v7->m128i_i64[0];
          *(_QWORD *)(a1 + 16) = v7[1].m128i_i64[0];
        }
        v8 = v7->m128i_i64[1];
        v7->m128i_i64[0] = (__int64)v7[1].m128i_i64;
        v9 = (_QWORD *)v18[0];
        v7->m128i_i64[1] = 0;
        *(_QWORD *)(a1 + 8) = v8;
        v7[1].m128i_i8[0] = 0;
        if ( v9 != v19 )
          j_j___libc_free_0(v9, v19[0] + 1LL);
        if ( v15 != v17 )
          j_j___libc_free_0(v15, v17[0] + 1LL);
      }
      return a1;
  }
}
