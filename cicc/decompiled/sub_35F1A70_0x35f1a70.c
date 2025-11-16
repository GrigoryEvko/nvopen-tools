// Function: sub_35F1A70
// Address: 0x35f1a70
//
__int64 __fastcall sub_35F1A70(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rdx
  _DWORD *v7; // rdx
  char *v8; // rsi
  __int64 result; // rax

  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
  v6 = *(_QWORD *)(a4 + 32);
  switch ( ((unsigned int)v4 >> 1) & 7 )
  {
    case 0u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".add", 4u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v6 = 1684300078;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
      goto LABEL_12;
    case 1u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".min", 4u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v6 = 1852402990;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
      goto LABEL_12;
    case 2u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".max", 4u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v6 = 2019650862;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
LABEL_12:
      v8 = ".s32";
      if ( (v4 & 1) != 0 )
        goto LABEL_6;
      if ( (((unsigned int)v4 >> 1) & 7) - 3 <= 2 )
        goto LABEL_5;
      v8 = ".u32";
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v7 > 3u )
        goto LABEL_15;
      return sub_CB6200(a4, (unsigned __int8 *)v8, 4u);
    case 3u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".and", 4u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v6 = 1684955438;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
      goto LABEL_4;
    case 4u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 2 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".or", 3u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_BYTE *)(v6 + 2) = 114;
        *(_WORD *)v6 = 28462;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 3LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
      goto LABEL_4;
    case 5u:
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v6) <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)".xor", 4u);
        v7 = *(_DWORD **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v6 = 1919907886;
        v7 = (_DWORD *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = v7;
      }
LABEL_4:
      v8 = ".s32";
      if ( (v4 & 1) == 0 )
LABEL_5:
        v8 = ".b32";
LABEL_6:
      if ( *(_QWORD *)(a4 + 24) - (_QWORD)v7 <= 3u )
        return sub_CB6200(a4, (unsigned __int8 *)v8, 4u);
LABEL_15:
      result = *(unsigned int *)v8;
      *v7 = result;
      *(_QWORD *)(a4 + 32) += 4LL;
      return result;
    default:
      BUG();
  }
}
