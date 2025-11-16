// Function: sub_E53640
// Address: 0xe53640
//
_BYTE *__fastcall sub_E53640(
        __int64 a1,
        int a2,
        unsigned int a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        char a7)
{
  char *v10; // r15
  __int64 v11; // r12
  __m128i *v12; // rdx
  size_t v13; // r9
  _QWORD *v14; // rax
  _WORD *v15; // rax
  __int64 v16; // rax
  _WORD *v17; // rdx
  __int64 v18; // rdi
  __int64 v20; // rax
  unsigned __int64 v21; // rsi
  char *v22; // rax
  char *v23; // r15
  unsigned int v24; // ecx
  unsigned int v25; // eax
  __int64 v26; // rdx

  switch ( a2 )
  {
    case 0:
      v10 = "unknown";
      break;
    case 1:
      v10 = "macos";
      break;
    case 2:
      v10 = "ios";
      break;
    case 3:
      v10 = "tvos";
      break;
    case 4:
      v10 = "watchos";
      break;
    case 5:
      v10 = "bridgeos";
      break;
    case 6:
      v10 = "macCatalyst";
      break;
    case 7:
      v10 = "iossimulator";
      break;
    case 8:
      v10 = "tvossimulator";
      break;
    case 9:
      v10 = "watchossimulator";
      break;
    case 10:
      v10 = "driverkit";
      break;
    case 11:
      v10 = "xros";
      break;
    case 12:
      v10 = "xrsimulator";
      break;
    default:
      BUG();
  }
  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(__m128i **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xFu )
  {
    v11 = sub_CB6200(*(_QWORD *)(a1 + 304), "\t.build_version ", 0x10u);
  }
  else
  {
    *v12 = _mm_load_si128((const __m128i *)&xmmword_3F7F8B0);
    *(_QWORD *)(v11 + 32) += 16LL;
  }
  v13 = strlen(v10);
  v14 = *(_QWORD **)(v11 + 32);
  if ( v13 > *(_QWORD *)(v11 + 24) - (_QWORD)v14 )
  {
    v11 = sub_CB6200(v11, (unsigned __int8 *)v10, v13);
    v15 = *(_WORD **)(v11 + 32);
    goto LABEL_7;
  }
  if ( (unsigned int)v13 >= 8 )
  {
    v21 = (unsigned __int64)(v14 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v14 = *(_QWORD *)v10;
    *(_QWORD *)((char *)v14 + (unsigned int)v13 - 8) = *(_QWORD *)&v10[(unsigned int)v13 - 8];
    v22 = (char *)v14 - v21;
    v23 = (char *)(v10 - v22);
    if ( (((_DWORD)v13 + (_DWORD)v22) & 0xFFFFFFF8) >= 8 )
    {
      v24 = (v13 + (_DWORD)v22) & 0xFFFFFFF8;
      v25 = 0;
      do
      {
        v26 = v25;
        v25 += 8;
        *(_QWORD *)(v21 + v26) = *(_QWORD *)&v23[v26];
      }
      while ( v25 < v24 );
    }
  }
  else
  {
    if ( (v13 & 4) != 0 )
    {
      *(_DWORD *)v14 = *(_DWORD *)v10;
      *(_DWORD *)((char *)v14 + (unsigned int)v13 - 4) = *(_DWORD *)&v10[(unsigned int)v13 - 4];
      v14 = *(_QWORD **)(v11 + 32);
      goto LABEL_34;
    }
    if ( !(_DWORD)v13 )
    {
LABEL_34:
      v15 = (_WORD *)((char *)v14 + v13);
      *(_QWORD *)(v11 + 32) = v15;
      goto LABEL_7;
    }
    *(_BYTE *)v14 = *v10;
    if ( (v13 & 2) != 0 )
    {
      *(_WORD *)((char *)v14 + (unsigned int)v13 - 2) = *(_WORD *)&v10[(unsigned int)v13 - 2];
      v14 = *(_QWORD **)(v11 + 32);
      goto LABEL_34;
    }
  }
  v15 = (_WORD *)(v13 + *(_QWORD *)(v11 + 32));
  *(_QWORD *)(v11 + 32) = v15;
LABEL_7:
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v15 <= 1u )
  {
    v11 = sub_CB6200(v11, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v15 = 8236;
    *(_QWORD *)(v11 + 32) += 2LL;
  }
  v16 = sub_CB59D0(v11, a3);
  v17 = *(_WORD **)(v16 + 32);
  v18 = v16;
  if ( *(_QWORD *)(v16 + 24) - (_QWORD)v17 <= 1u )
  {
    v18 = sub_CB6200(v16, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v17 = 8236;
    *(_QWORD *)(v16 + 32) += 2LL;
  }
  sub_CB59D0(v18, a4);
  if ( a5 )
  {
    v20 = sub_904010(*(_QWORD *)(a1 + 304), ", ");
    sub_CB59D0(v20, a5);
  }
  sub_E534F0(*(_QWORD *)(a1 + 304), &a7);
  return sub_E4D880(a1);
}
