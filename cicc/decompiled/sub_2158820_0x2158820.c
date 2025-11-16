// Function: sub_2158820
// Address: 0x2158820
//
__int64 __fastcall sub_2158820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  size_t v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  char v13; // al
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // rax

  switch ( *(_BYTE *)a2 )
  {
    case 0:
      v9 = sub_21583D0(a1, *(_DWORD *)(a2 + 8));
      *(_BYTE *)a3 = 1;
      *(_QWORD *)(a3 + 8) = _mm_cvtsi32_si128(v9).m128i_u64[0];
      result = 1;
      break;
    case 1:
      v10 = *(_QWORD *)(a2 + 24);
      *(_BYTE *)a3 = 2;
      *(_QWORD *)(a3 + 8) = v10;
      result = 1;
      break;
    case 2:
    case 5:
    case 6:
    case 7:
    case 8:
    case 0xA:
      v6 = sub_396EAF0(a1, *(_QWORD *)(a2 + 24));
      goto LABEL_5;
    case 3:
      v11 = *(_QWORD *)(a2 + 24);
      v12 = v11 + 24;
      v13 = *(_BYTE *)(*(_QWORD *)v11 + 8LL);
      switch ( v13 )
      {
        case 2:
          v14 = *(_QWORD *)(a1 + 248);
          v15 = 2;
          break;
        case 3:
          v14 = *(_QWORD *)(a1 + 248);
          v15 = 3;
          break;
        case 1:
          v14 = *(_QWORD *)(a1 + 248);
          v15 = 1;
          break;
        default:
          sub_16BD130("Unsupported FP type", 1u);
      }
      v16 = sub_2163260(v15, v12, v14);
      *(_BYTE *)a3 = 4;
      v17 = v16;
      v18 = 0;
      if ( v17 )
        v18 = v17 + 8;
      *(_QWORD *)(a3 + 8) = v18;
      result = 1;
      break;
    case 4:
      v19 = *(_QWORD *)(a1 + 248);
      v20 = sub_1DD5A70(*(_QWORD *)(a2 + 24));
      v21 = sub_38CF310(v20, 0, v19, 0);
      *(_BYTE *)a3 = 4;
      *(_QWORD *)(a3 + 8) = v21;
      result = 1;
      break;
    case 9:
      v4 = *(_QWORD *)(a2 + 24);
      v5 = 0;
      if ( v4 )
        v5 = strlen(*(const char **)(a2 + 24));
      v6 = sub_3970BC0(a1, v4, v5);
LABEL_5:
      v7 = sub_38CF310(v6, 0, *(_QWORD *)(a1 + 248), 0);
      *(_BYTE *)a3 = 4;
      *(_QWORD *)(a3 + 8) = v7;
      result = 1;
      break;
  }
  return result;
}
