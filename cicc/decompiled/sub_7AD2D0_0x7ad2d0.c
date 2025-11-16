// Function: sub_7AD2D0
// Address: 0x7ad2d0
//
void __fastcall sub_7AD2D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  int v5; // edx
  unsigned int v6; // eax
  __int64 v8; // r12
  __int64 v9; // rdi
  __int16 v10; // cx
  int v11; // edi
  __int16 v12; // si
  __int64 v13; // rax
  char v14; // dl
  char *v15; // rdi

  v5 = dword_4F08488;
  v6 = *(_DWORD *)(a1 + 8);
  v8 = *(unsigned __int16 *)(a1 + 24);
  if ( v6 > dword_4F08488 )
  {
    v10 = *(_WORD *)(a1 + 12);
    dword_4F08488 = *(_DWORD *)(a1 + 8);
    v11 = v6 - v5;
    v12 = v10 - 1;
LABEL_24:
    sub_7AD240(v11, v12);
    goto LABEL_4;
  }
  if ( (v8 & 0xFFF7) != 0x43 && unk_4F06C40 )
  {
    v12 = 1;
    v11 = 0;
    goto LABEL_24;
  }
LABEL_4:
  if ( (_WORD)v8 == 10 )
    return;
  if ( *(_BYTE *)(a1 + 26) == 4 )
  {
    sub_7295A0(*(char **)(a1 + 48));
    return;
  }
  if ( (unsigned __int16)(v8 - 4) <= 4u || (_WORD)v8 == 2 )
  {
    v9 = *(_QWORD *)(a1 + 48);
    if ( (_WORD)v8 == 8 )
    {
      if ( !v9 || *(_BYTE *)(v9 + 173) )
        goto LABEL_29;
    }
    else if ( !v9 || *(_BYTE *)(v9 + 173) )
    {
      goto LABEL_20;
    }
    byte_4F08468 = 0;
    if ( (_WORD)v8 != 8 )
    {
LABEL_20:
      sub_748000(v9, 1, (__int64)&qword_4F083E0, a4, a5);
LABEL_21:
      byte_4F08468 = 1;
      return;
    }
LABEL_29:
    v13 = *(_QWORD *)(a1 + 64);
    if ( v13
      && ((v14 = *(_BYTE *)(v13 + 80), v14 == 20) || v14 == 11 && (*(_BYTE *)(*(_QWORD *)(v13 + 88) + 207LL) & 1) != 0) )
    {
      sub_7295A0(*(char **)(*(_QWORD *)(a1 + 56) + 184LL));
    }
    else
    {
      byte_4F0847D = 1;
      sub_748000(v9, 0, (__int64)&qword_4F083E0, a4, a5);
      byte_4F0847D = 0;
    }
    sub_7295A0(*(char **)(a1 + 72));
    if ( byte_4F0847E )
    {
      sub_729660(41);
      byte_4F0847E = 0;
    }
    goto LABEL_21;
  }
  if ( (_WORD)v8 == 1 )
  {
    sub_7295A0(*(char **)(*(_QWORD *)(a1 + 48) + 8LL));
  }
  else if ( (unsigned __int16)(v8 - 118) <= 1u )
  {
    v15 = "restrict";
    if ( dword_4F068C4 )
      v15 = "__restrict__";
    sub_7295A0(v15);
  }
  else
  {
    if ( !HIDWORD(qword_4F077B4) )
    {
LABEL_16:
      sub_7295A0((char *)*(&off_4B6DFA0 + v8));
      return;
    }
    switch ( (_WORD)v8 )
    {
      case 0x95:
        sub_7295A0("__asm__");
        break;
      case 0x71:
        sub_7295A0("__builtin_va_start");
        break;
      case 0x72:
        sub_7295A0("__builtin_va_arg");
        break;
      case 0x73:
        sub_7295A0("__builtin_va_end");
        break;
      case 0x74:
        sub_7295A0("__builtin_va_copy");
        break;
      default:
        goto LABEL_16;
    }
  }
}
