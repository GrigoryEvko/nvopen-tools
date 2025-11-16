// Function: sub_21E6B10
// Address: 0x21e6b10
//
__int64 __fastcall sub_21E6B10(__int64 a1, unsigned int a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // rdx
  unsigned __int64 v7; // rdi
  char *v8; // rsi
  __int64 result; // rax

  v4 = *(_QWORD *)(a3 + 16);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL * a2 + 8);
  v6 = *(_QWORD *)(a3 + 24);
  v7 = v4 - v6;
  switch ( ((unsigned int)v5 >> 1) & 7 )
  {
    case 0u:
      if ( v7 <= 3 )
      {
        sub_16E7EE0(a3, ".add", 4u);
        v6 = *(_QWORD *)(a3 + 24);
        v4 = *(_QWORD *)(a3 + 16);
      }
      else
      {
        *(_DWORD *)v6 = 1684300078;
        v6 = *(_QWORD *)(a3 + 24) + 4LL;
        v4 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a3 + 24) = v6;
      }
      goto LABEL_3;
    case 1u:
      if ( v7 <= 3 )
      {
        sub_16E7EE0(a3, ".min", 4u);
        v6 = *(_QWORD *)(a3 + 24);
        v4 = *(_QWORD *)(a3 + 16);
      }
      else
      {
        *(_DWORD *)v6 = 1852402990;
        v6 = *(_QWORD *)(a3 + 24) + 4LL;
        v4 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a3 + 24) = v6;
      }
      goto LABEL_3;
    case 2u:
      if ( v7 <= 3 )
      {
        sub_16E7EE0(a3, ".max", 4u);
        v6 = *(_QWORD *)(a3 + 24);
        v4 = *(_QWORD *)(a3 + 16);
      }
      else
      {
        *(_DWORD *)v6 = 2019650862;
        v6 = *(_QWORD *)(a3 + 24) + 4LL;
        v4 = *(_QWORD *)(a3 + 16);
        *(_QWORD *)(a3 + 24) = v6;
      }
      goto LABEL_3;
    case 3u:
      if ( v7 <= 3 )
      {
        sub_16E7EE0(a3, ".and", 4u);
        v6 = *(_QWORD *)(a3 + 24);
      }
      else
      {
        *(_DWORD *)v6 = 1684955438;
        v6 = *(_QWORD *)(a3 + 24) + 4LL;
        *(_QWORD *)(a3 + 24) = v6;
      }
      goto LABEL_12;
    case 4u:
      if ( v7 <= 2 )
      {
        sub_16E7EE0(a3, ".or", 3u);
        v6 = *(_QWORD *)(a3 + 24);
      }
      else
      {
        *(_BYTE *)(v6 + 2) = 114;
        *(_WORD *)v6 = 28462;
        v6 = *(_QWORD *)(a3 + 24) + 3LL;
        *(_QWORD *)(a3 + 24) = v6;
      }
      goto LABEL_12;
    case 5u:
      if ( v7 <= 3 )
      {
        sub_16E7EE0(a3, ".xor", 4u);
        v6 = *(_QWORD *)(a3 + 24);
      }
      else
      {
        *(_DWORD *)v6 = 1919907886;
        v6 = *(_QWORD *)(a3 + 24) + 4LL;
        *(_QWORD *)(a3 + 24) = v6;
      }
LABEL_12:
      if ( (v5 & 1) == 0 )
      {
        v4 = *(_QWORD *)(a3 + 16);
        v8 = ".b32";
LABEL_8:
        if ( (unsigned __int64)(v4 - v6) > 3 )
          goto LABEL_9;
        return sub_16E7EE0(a3, v8, 4u);
      }
      v4 = *(_QWORD *)(a3 + 16);
      v8 = ".s32";
LABEL_4:
      if ( (unsigned __int64)(v4 - v6) <= 3 )
        return sub_16E7EE0(a3, v8, 4u);
LABEL_9:
      result = *(unsigned int *)v8;
      *(_DWORD *)v6 = result;
      *(_QWORD *)(a3 + 24) += 4LL;
      return result;
    default:
LABEL_3:
      v8 = ".s32";
      if ( (v5 & 1) != 0 )
        goto LABEL_4;
      v8 = ".u32";
      if ( (((unsigned int)v5 >> 1) & 7) - 3 > 2 )
        goto LABEL_4;
      v8 = ".b32";
      goto LABEL_8;
  }
}
