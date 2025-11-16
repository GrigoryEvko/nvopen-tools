// Function: sub_74D000
// Address: 0x74d000
//
void __fastcall sub_74D000(__int64 a1, __int64 a2)
{
  void (__fastcall *v3)(__int64, __int64); // rax
  char v4; // al
  unsigned __int8 v5; // al
  char *v6; // rdi

  v3 = *(void (__fastcall **)(__int64, __int64))(a2 + 24);
  if ( v3 )
  {
    v3(a1, 6);
    return;
  }
  if ( dword_4F072C8 == 1 )
  {
    v5 = *(_BYTE *)(a1 + 140);
  }
  else
  {
    v4 = *(_BYTE *)(a1 + 89);
    if ( (v4 & 0x40) == 0 )
    {
      if ( (v4 & 8) != 0 ? *(_QWORD *)(a1 + 24) : *(_QWORD *)(a1 + 8) )
        goto LABEL_10;
    }
    v5 = *(_BYTE *)(a1 + 140);
    if ( v5 == 9 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(a1 + 168) + 109LL) & 0x20) != 0 )
        goto LABEL_10;
      goto LABEL_16;
    }
  }
  if ( v5 == 10 )
  {
    v6 = "struct";
    goto LABEL_9;
  }
  if ( v5 > 0xAu )
  {
    if ( v5 == 11 )
    {
      v6 = "union";
      goto LABEL_9;
    }
    goto LABEL_24;
  }
  v6 = "enum";
  if ( v5 != 2 )
  {
    if ( v5 == 9 )
    {
LABEL_16:
      v6 = "class";
      goto LABEL_9;
    }
LABEL_24:
    sub_721090();
  }
LABEL_9:
  (*(void (__fastcall **)(char *, __int64))a2)(v6, a2);
  (*(void (__fastcall **)(char *, __int64))a2)(" ", a2);
LABEL_10:
  sub_74C550(a1, 6u, a2);
}
