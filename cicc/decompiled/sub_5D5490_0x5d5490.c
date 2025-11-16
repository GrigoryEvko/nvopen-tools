// Function: sub_5D5490
// Address: 0x5d5490
//
void __fastcall sub_5D5490(__int64 a1)
{
  __int64 *v1; // rcx
  __int64 *v2; // rax
  char v3; // dl
  __int64 v4; // r13
  char v5; // al
  char v6; // al

  if ( *(_BYTE *)(a1 + 24) == 1 && *(_BYTE *)(a1 + 56) == 94 )
  {
    v1 = *(__int64 **)(a1 + 72);
    v2 = v1;
    if ( (*((_BYTE *)v1 + 25) & 1) == 0 )
    {
      v5 = *((_BYTE *)v1 + 24);
      if ( v5 == 3 )
        return;
      if ( v5 == 1 )
      {
        v6 = *((_BYTE *)v1 + 56);
        if ( v6 == 91 )
        {
          if ( (*(_BYTE *)(*(_QWORD *)(v1[9] + 16) + 25LL) & 1) != 0 )
            return;
        }
        else if ( v6 == 3 )
        {
          return;
        }
      }
LABEL_10:
      v4 = *v1;
      sub_74A390(*v1, 0, 1, 0, 0, &qword_4CF7CE0);
      sub_5D34A0();
      sub_74D110(v4, 0, 0, &qword_4CF7CE0);
      putc(59, stream);
      ++dword_4CF7F40;
      return;
    }
    while ( *((_BYTE *)v2 + 24) == 1 )
    {
      v3 = *((_BYTE *)v2 + 56);
      if ( v3 == 91 )
        goto LABEL_10;
      if ( v3 != 94 )
        return;
      v2 = (__int64 *)v2[9];
    }
  }
}
