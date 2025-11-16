// Function: sub_8C2270
// Address: 0x8c2270
//
__int64 __fastcall sub_8C2270(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  char v14; // al
  __int64 v15; // rcx
  int v16; // edx
  __int64 v17; // rdx
  __m128i *v18; // rbx
  __int64 v19; // r14
  char v20; // dl
  __int64 v21; // rbx
  _BOOL4 v22; // ebx
  __int64 v23; // r12
  int v24; // edx
  __int64 v25; // [rsp+0h] [rbp-40h]
  __int64 v26; // [rsp+0h] [rbp-40h]
  int v27; // [rsp+Ch] [rbp-34h]
  int v28; // [rsp+Ch] [rbp-34h]

  result = *(unsigned __int8 *)(a1 + 80);
  switch ( (char)result )
  {
    case 4:
    case 5:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
      break;
    case 6:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v3 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v3 = *(_QWORD *)(a1 + 88);
      break;
    default:
      BUG();
  }
  if ( (*(_BYTE *)(v3 + 267) & 1) == 0 )
    goto LABEL_9;
  v4 = *(_QWORD *)(a1 + 88);
  v5 = *(_QWORD *)(v4 + 88);
  if ( v5 )
  {
    if ( (*(_BYTE *)(v4 + 160) & 1) != 0 )
      v5 = a1;
  }
  else
  {
    v5 = a1;
  }
  if ( (*(_BYTE *)(v5 + 81) & 2) != 0 && (*(_BYTE *)(v3 + 267) & 2) != 0 )
  {
LABEL_9:
    if ( (*(_BYTE *)(v3 + 265) & 1) == 0 )
    {
      v17 = *(_QWORD *)(a1 + 88);
      v18 = *(__m128i **)(v17 + 88);
      if ( v18 && (*(_BYTE *)(v17 + 160) & 1) == 0 )
        LOBYTE(result) = v18[5].m128i_i8[0];
      else
        v18 = (__m128i *)a1;
      switch ( (char)result )
      {
        case 4:
        case 5:
          v6 = *(_QWORD *)(v18[6].m128i_i64[0] + 80);
          break;
        case 6:
          v6 = *(_QWORD *)(v18[6].m128i_i64[0] + 32);
          break;
        case 9:
        case 10:
          v6 = *(_QWORD *)(v18[6].m128i_i64[0] + 56);
          break;
        case 19:
        case 20:
        case 21:
        case 22:
          v6 = v18[5].m128i_i64[1];
          break;
        default:
          BUG();
      }
      v7 = *(_QWORD *)(v6 + 176);
      v8 = *(_QWORD *)(v7 + 88);
      v9 = *(_QWORD *)(*(_QWORD *)(v7 + 96) + 8LL);
      v25 = v9;
      if ( v9 )
      {
        if ( *(_BYTE *)(v9 + 80) == 17 )
        {
          v27 = 1;
          v10 = *(_QWORD *)(v9 + 88);
        }
        else
        {
          v27 = 0;
          v10 = v9;
        }
      }
      else
      {
        v27 = 0;
        v10 = 0;
      }
      if ( (*(_BYTE *)(v17 + 267) & 2) != 0 )
      {
        sub_890550(a1);
      }
      else
      {
        v11 = 0;
        if ( v8 )
        {
          v12 = sub_724EF0(v8);
          *((_DWORD *)v12 + 9) = 1;
          v11 = (__int64)v12;
        }
        sub_8C2140(a1, v8, v11);
      }
      for ( ; v10; v10 = *(_QWORD *)(v10 + 8) )
      {
        v13 = sub_8C18E0(v18, v8, v10, a1);
        if ( v13 )
          sub_87F0B0((__int64)v13, (__int64 *)(*(_QWORD *)(a1 + 88) + 216LL));
        if ( !v27 )
          break;
      }
      if ( !v25 )
        sub_8C2140(a1, v8, 0);
      v14 = sub_8D23B0(v8);
      v15 = *(_QWORD *)(a1 + 88);
      v16 = 2 * (v14 & 1);
      result = v16 | *(_BYTE *)(v15 + 267) & 0xFDu;
      *(_BYTE *)(v15 + 267) = v16 | *(_BYTE *)(v15 + 267) & 0xFD;
      goto LABEL_28;
    }
    switch ( (char)result )
    {
      case 4:
      case 5:
        v19 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 80LL);
        break;
      case 6:
        v19 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 32LL);
        break;
      case 9:
      case 10:
        v19 = *(_QWORD *)(*(_QWORD *)(a1 + 96) + 56LL);
        break;
      case 19:
      case 20:
      case 21:
      case 22:
        v19 = *(_QWORD *)(a1 + 88);
        break;
      default:
        BUG();
    }
    result = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v19 + 176) + 88LL) + 160LL);
    v20 = *(_BYTE *)(result + 140);
    if ( (unsigned __int8)(v20 - 9) > 2u )
    {
      if ( v20 != 12 || *(_BYTE *)(result + 184) != 10 )
        goto LABEL_28;
      result = *(_QWORD *)(*(_QWORD *)(result + 168) + 16LL);
      v21 = *(_QWORD *)result;
    }
    else
    {
      if ( (*(_BYTE *)(result + 177) & 0x10) == 0 )
        goto LABEL_28;
      result = *(_QWORD *)(*(_QWORD *)result + 96LL);
      v21 = *(_QWORD *)(result + 72);
    }
    if ( !v21 || (*(_BYTE *)(v21 + 84) & 2) != 0 )
      goto LABEL_28;
    sub_8C2270(v21);
    switch ( *(_BYTE *)(v21 + 80) )
    {
      case 4:
      case 5:
        v26 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 80LL);
        break;
      case 6:
        v26 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v26 = *(_QWORD *)(*(_QWORD *)(v21 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v26 = *(_QWORD *)(v21 + 88);
        break;
      default:
        BUG();
    }
    result = v26;
    v22 = (*(_BYTE *)(v19 + 267) & 2) != 0;
    if ( (*(_BYTE *)(v26 + 267) & 2) == 0 )
    {
      if ( (*(_BYTE *)(v19 + 267) & 2) != 0 )
        sub_890550(a1);
LABEL_61:
      v23 = *(_QWORD *)(v26 + 216);
      if ( v23 )
      {
        v28 = 0;
        if ( *(_BYTE *)(v23 + 80) != 17 )
        {
          do
          {
LABEL_63:
            if ( !v22 || *(_BYTE *)(v23 + 80) != 20 || *(_QWORD *)(*(_QWORD *)(v23 + 88) + 416LL) )
              sub_8C0950(a1, (__int64 *)v23);
            if ( !v28 )
              break;
            v23 = *(_QWORD *)(v23 + 8);
          }
          while ( v23 );
          goto LABEL_68;
        }
        v23 = *(_QWORD *)(v23 + 88);
        if ( v23 )
        {
          v28 = 1;
          goto LABEL_63;
        }
      }
LABEL_68:
      v24 = *(_BYTE *)(v26 + 267) & 2;
      result = v24 | *(_BYTE *)(v19 + 267) & 0xFDu;
      *(_BYTE *)(v19 + 267) = v24 | *(_BYTE *)(v19 + 267) & 0xFD;
      goto LABEL_28;
    }
    if ( (*(_BYTE *)(v19 + 267) & 2) == 0 )
      goto LABEL_61;
LABEL_28:
    *(_BYTE *)(v3 + 267) |= 1u;
  }
  return result;
}
