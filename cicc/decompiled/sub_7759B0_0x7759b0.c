// Function: sub_7759B0
// Address: 0x7759b0
//
__int64 __fastcall sub_7759B0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, _BOOL4 *a5)
{
  __int64 result; // rax
  const __m128i *v10; // rsi
  __int64 v11; // rdi
  const __m128i *v12; // rdx
  _BOOL4 v13; // eax
  int v14; // r8d
  __int64 v15; // rax
  __int8 v16; // al
  __int64 v17; // rax
  char v18; // dl
  __int64 v19; // rax
  _DWORD v20[13]; // [rsp+1Ch] [rbp-34h] BYREF

  switch ( *(_BYTE *)(a4 + 140) )
  {
    case 2:
      v15 = sub_620EE0(a2, byte_4B6DF90[*(unsigned __int8 *)(a4 + 160)], v20);
      *a5 = v15 != 0 || v20[0] != 0;
      return 1;
    case 3:
    case 4:
      v10 = a2;
      LOBYTE(v11) = *(_BYTE *)(a4 + 160);
      v12 = &xmmword_4F081A0[(unsigned __int8)v11];
      goto LABEL_8;
    case 5:
      v14 = sub_70BE30(*(_BYTE *)(a4 + 160), a2, &xmmword_4F081A0[*(unsigned __int8 *)(a4 + 160)], v20);
      v13 = 1;
      if ( !v14 )
      {
        v10 = a2 + 1;
        v11 = *(unsigned __int8 *)(a4 + 160);
        v12 = &xmmword_4F081A0[v11];
LABEL_8:
        v13 = sub_70BE30(v11, v10, v12, v20) != 0;
      }
      *a5 = v13;
      return 1;
    case 6:
      v16 = a2->m128i_i8[8];
      if ( (v16 & 1) == 0 )
      {
        if ( (v16 & 0x20) != 0 || a2->m128i_i64[0] )
          goto LABEL_22;
        goto LABEL_4;
      }
      v17 = a2[1].m128i_i64[0];
      v18 = *(_BYTE *)(v17 + 173);
      if ( v18 == 1 )
      {
        if ( (unsigned int)sub_621000((__int16 *)(v17 + 176), 0, (__int16 *)&xmmword_4F08290, 0) )
        {
LABEL_22:
          *a5 = 1;
          return 1;
        }
LABEL_4:
        *a5 = 0;
        return 1;
      }
      if ( v18 == 6 && *(_BYTE *)(v17 + 176) == 1 )
      {
        v19 = *(_QWORD *)(v17 + 184);
        if ( v19 )
        {
          if ( (*(_BYTE *)(v19 + 168) & 8) == 0 )
            goto LABEL_22;
        }
      }
      *a5 = 0;
      result = 0;
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xA8Du, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return 0;
      }
      return result;
    case 0xD:
      if ( a2->m128i_i64[1] )
        goto LABEL_22;
      goto LABEL_4;
    case 0x13:
      goto LABEL_4;
    default:
      *a5 = 1;
      return 0;
  }
}
