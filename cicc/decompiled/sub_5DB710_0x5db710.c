// Function: sub_5DB710
// Address: 0x5db710
//
int __fastcall sub_5DB710(__int64 a1)
{
  FILE *v2; // rsi
  char v3; // al
  int v4; // r13d
  char *v5; // rdx
  int v6; // edi
  char *v7; // rbx
  int result; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  int v13; // edi
  char *v14; // rbx

  if ( *(char *)(a1 + 141) < 0 && !(unsigned int)sub_8D23B0(a1) )
  {
    if ( *(_BYTE *)(a1 + 140) == 2 )
      return sub_5D72F0(a1, 0, v9, v10, v11, v12);
    if ( (*(_BYTE *)(a1 + 178) & 0x40) != 0 )
      return sub_5DAD30((const char *)a1, 0);
  }
  if ( !unk_4F068C4
    || (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u
    || !(unk_4F06A80 | unk_4F06A7C | unk_4F06A78)
    || (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 110LL) & 0x40) == 0 )
  {
    v2 = stream;
    if ( stream == qword_4CF7EB0 )
    {
      if ( (*(_BYTE *)(a1 + 141) & 4) != 0 )
      {
LABEL_13:
        putc(32, v2);
        ++dword_4CF7F40;
        return sub_5D71E0(a1);
      }
      v3 = *(_BYTE *)(a1 + 140);
      if ( v3 != 10 )
      {
LABEL_8:
        v4 = 5;
        v5 = "union";
        if ( v3 != 11 )
        {
          if ( v3 != 2 )
            sub_721090(a1);
          v4 = 4;
          v6 = 101;
          v7 = "num";
          goto LABEL_11;
        }
LABEL_9:
        v6 = *v5;
        v7 = v5 + 1;
LABEL_11:
        while ( 1 )
        {
          putc(v6, v2);
          v6 = *v7++;
          if ( !(_BYTE)v6 )
            break;
          v2 = stream;
        }
        dword_4CF7F40 += v4;
        v2 = stream;
        goto LABEL_13;
      }
    }
    else
    {
      v3 = *(_BYTE *)(a1 + 140);
      if ( v3 != 10 )
        goto LABEL_8;
    }
    v4 = 6;
    v5 = "struct";
    goto LABEL_9;
  }
  v13 = 95;
  v14 = "_va_list_tag_type";
  do
  {
    ++v14;
    result = putc(v13, stream);
    v13 = *(v14 - 1);
  }
  while ( *(v14 - 1) );
  dword_4CF7F40 += 18;
  return result;
}
