// Function: sub_747C50
// Address: 0x747c50
//
void __fastcall sub_747C50(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, _QWORD); // rax
  char v4; // r13
  void (__fastcall *v5)(const char *); // rax
  __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rdi
  char v10; // al
  __int64 v11; // rax
  char v12; // r14
  __int64 v13; // r15
  void (__fastcall *v14)(const char *); // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // r14
  char v19; // dl
  char v20; // [rsp+0h] [rbp-70h] BYREF
  char v21; // [rsp+1h] [rbp-6Fh]

  v3 = *(void (__fastcall **)(__int64, _QWORD))(a2 + 72);
  v4 = *(_BYTE *)(a2 + 149);
  *(_BYTE *)(a2 + 149) = 0;
  if ( v3 )
  {
    v3(a1, 0);
  }
  else if ( a1 )
  {
    switch ( *(_BYTE *)(a1 + 24) )
    {
      case 0:
        (*(void (__fastcall **)(const char *))a2)("<error>");
        break;
      case 1:
        v12 = *(_BYTE *)(a1 + 56);
        v13 = *(_QWORD *)(a1 + 72);
        if ( v12 == 25 )
        {
          (*(void (__fastcall **)(char *))a2)("(");
          sub_747C50(v13, a2);
          (*(void (__fastcall **)(char *, __int64))a2)(")", a2);
        }
        else if ( (unsigned __int8)(v12 - 8) <= 1u
               || ((v12 - 13) & 0xF7) == 0
               || (*(_BYTE *)(a1 + 27) & 2) != 0 && sub_730740(a1) )
        {
          sub_747C50(v13, a2);
        }
        else
        {
          v14 = *(void (__fastcall **)(const char *))a2;
          if ( !v12
            && *(_BYTE *)(v13 + 24) == 1
            && *(_BYTE *)(v13 + 56) == 116
            && (v17 = *(_QWORD *)(v13 + 72), *(_BYTE *)(v17 + 24) == 2)
            && (v18 = *(_QWORD *)(v17 + 56), *(_BYTE *)(v18 + 173) == 12)
            && ((v19 = *(_BYTE *)(v18 + 176), (unsigned __int8)(v19 - 2) <= 1u) || v19 == 11) )
          {
            ((void (__fastcall *)(char *, __int64))v14)("&", a2);
            sub_747A20(v18, a2);
          }
          else
          {
            ((void (__fastcall *)(const char *, __int64))v14)("<expression>", a2);
          }
        }
        break;
      case 2:
        sub_748000(*(_QWORD *)(a1 + 56), 1, a2);
        break;
      case 3:
        sub_74C550(*(_QWORD *)(a1 + 56), 7, a2);
        break;
      case 4:
        sub_74C550(*(_QWORD *)(a1 + 56), 8, a2);
        break;
      case 5:
        v9 = *(_QWORD *)a1;
        v10 = *(_BYTE *)(v9 + 89);
        if ( (v10 & 0x40) != 0 )
          goto LABEL_6;
        v11 = (v10 & 8) != 0 ? *(_QWORD *)(v9 + 24) : *(_QWORD *)(v9 + 8);
        if ( !v11 || *(_BYTE *)(*(_QWORD *)(a1 + 56) + 48LL) > 1u )
          goto LABEL_6;
        sub_74B930(v9, a2);
        (*(void (__fastcall **)(char *, __int64))a2)("()", a2);
        break;
      case 0x14:
        v8 = *(_QWORD *)(a1 + 56);
        if ( v8 )
          sub_74C550(v8, 11, a2);
        else
          (*(void (__fastcall **)(const char *))a2)("<NULL routine>");
        break;
      case 0x16:
        v7 = *(_QWORD *)(a1 + 56);
        if ( v7 )
          sub_74B930(v7, a2);
        else
          (*(void (__fastcall **)(const char *))a2)("<default>");
        break;
      case 0x18:
        v5 = *(void (__fastcall **)(const char *))a2;
        if ( *(_DWORD *)(a1 + 56) )
        {
          v5("<parameter #");
          v15 = *(unsigned int *)(a1 + 56);
          if ( (unsigned int)v15 > 9 )
          {
            sub_622470(v15, &v20);
          }
          else
          {
            v21 = 0;
            v20 = v15 + 48;
          }
          (*(void (__fastcall **)(char *, __int64))a2)(&v20, a2);
          if ( *(_DWORD *)(a1 + 60) == 2 )
          {
            (*(void (__fastcall **)(const char *, __int64))a2)(" (one level up)", a2);
          }
          else if ( *(_DWORD *)(a1 + 60) > 2u )
          {
            (*(void (__fastcall **)(char *, __int64))a2)(" (", a2);
            v16 = *(unsigned int *)(a1 + 60) - 1LL;
            if ( v16 > 9 )
            {
              sub_622470(v16, &v20);
            }
            else
            {
              v21 = 0;
              v20 = v16 + 48;
            }
            (*(void (__fastcall **)(char *, __int64))a2)(&v20, a2);
            (*(void (__fastcall **)(const char *, __int64))a2)(" levels up)", a2);
          }
          (*(void (__fastcall **)(char *, __int64))a2)(">", a2);
        }
        else
        {
          v5("this");
        }
        break;
      case 0x19:
        (*(void (__fastcall **)(const char *))a2)("{ ... }");
        break;
      case 0x20:
        v6 = *(_QWORD *)(a1 + 64);
        sub_74C550(*(_QWORD *)(a1 + 56), 59, a2);
        if ( v6 )
          (*(void (__fastcall **)(const char *, __int64))a2)("<...>", a2);
        break;
      default:
LABEL_6:
        (*(void (__fastcall **)(const char *, __int64))a2)("<expression>", a2);
        break;
    }
    if ( (*(_BYTE *)(a1 + 26) & 4) != 0 )
      (*(void (__fastcall **)(char *, __int64))a2)("...", a2);
  }
  else
  {
    (*(void (__fastcall **)(const char *))a2)("<NULL expression>");
  }
  *(_BYTE *)(a2 + 149) = v4;
}
