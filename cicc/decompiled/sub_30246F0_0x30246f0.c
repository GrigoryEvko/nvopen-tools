// Function: sub_30246F0
// Address: 0x30246f0
//
__int64 __fastcall sub_30246F0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  bool v7; // zf
  unsigned __int64 v8; // rcx
  char *v9; // r8
  __m128i *v10; // rax
  __int64 v11; // rcx
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rax
  char v14; // [rsp+14h] [rbp-6Ch] BYREF
  _BYTE v15[11]; // [rsp+15h] [rbp-6Bh] BYREF
  _BYTE *v16; // [rsp+20h] [rbp-60h]
  size_t v17; // [rsp+28h] [rbp-58h]
  _WORD v18[8]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v19[2]; // [rsp+40h] [rbp-40h] BYREF
  _QWORD v20[6]; // [rsp+50h] [rbp-30h] BYREF

  switch ( *(_BYTE *)(a3 + 8) )
  {
    case 0:
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
    case 0xC:
      v8 = *(_DWORD *)(a3 + 8) >> 8;
      if ( (_DWORD)v8 == 1 )
      {
        strcpy((char *)(a1 + 16), "pred");
        *(_QWORD *)a1 = a1 + 16;
        *(_QWORD *)(a1 + 8) = 4;
      }
      else
      {
        if ( (unsigned int)v8 > 0x40 )
          BUG();
        v17 = 1;
        v16 = v18;
        v18[0] = 117;
        if ( (_DWORD)v8 )
        {
          v9 = v15;
          do
          {
            *--v9 = v8 % 0xA + 48;
            v13 = v8;
            v8 /= 0xAu;
          }
          while ( v13 > 9 );
        }
        else
        {
          v14 = 48;
          v9 = &v14;
        }
        v19[0] = (__int64)v20;
        sub_3020560(v19, v9, (__int64)v15);
        v10 = (__m128i *)sub_2241130((unsigned __int64 *)v19, 0, 0, v16, v17);
        *(_QWORD *)a1 = a1 + 16;
        if ( (__m128i *)v10->m128i_i64[0] == &v10[1] )
        {
          *(__m128i *)(a1 + 16) = _mm_loadu_si128(v10 + 1);
        }
        else
        {
          *(_QWORD *)a1 = v10->m128i_i64[0];
          *(_QWORD *)(a1 + 16) = v10[1].m128i_i64[0];
        }
        v11 = v10->m128i_i64[1];
        v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
        v12 = (_QWORD *)v19[0];
        v10->m128i_i64[1] = 0;
        *(_QWORD *)(a1 + 8) = v11;
        v10[1].m128i_i8[0] = 0;
        if ( v12 != v20 )
          j_j___libc_free_0((unsigned __int64)v12);
        if ( v16 != (_BYTE *)v18 )
          j_j___libc_free_0((unsigned __int64)v16);
      }
      return a1;
    case 0xE:
      v7 = sub_AE2980(*(_QWORD *)(a2 + 200) + 16LL, *(_DWORD *)(a3 + 8) >> 8)[1] == 64;
      *(_QWORD *)a1 = a1 + 16;
      if ( v7 )
      {
        if ( a4 )
          *(_WORD *)(a1 + 16) = 13922;
        else
          *(_WORD *)(a1 + 16) = 13941;
        *(_BYTE *)(a1 + 18) = 52;
        *(_QWORD *)(a1 + 8) = 3;
        *(_BYTE *)(a1 + 19) = 0;
      }
      else
      {
        if ( a4 )
          *(_WORD *)(a1 + 16) = 13154;
        else
          *(_WORD *)(a1 + 16) = 13173;
        *(_BYTE *)(a1 + 18) = 50;
        *(_QWORD *)(a1 + 8) = 3;
        *(_BYTE *)(a1 + 19) = 0;
      }
      return a1;
    default:
      BUG();
  }
}
