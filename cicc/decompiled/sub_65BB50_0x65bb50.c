// Function: sub_65BB50
// Address: 0x65bb50
//
void __fastcall sub_65BB50(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // rcx
  char v4; // al
  __int64 v5; // rdx
  __int64 v6; // rax
  char v7; // dl
  __int64 v8; // rax

  v2 = *(_QWORD *)(a1 + 288);
  if ( *(_BYTE *)(v2 + 140) == 7 )
  {
    v3 = *(_QWORD *)(v2 + 160);
    v4 = *(_BYTE *)(v3 + 140);
    if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
    {
      if ( v4 == 12 )
      {
        v6 = *(_QWORD *)(v2 + 160);
        do
        {
          v6 = *(_QWORD *)(v6 + 160);
          v7 = *(_BYTE *)(v6 + 140);
        }
        while ( v7 == 12 );
        if ( v7 )
        {
LABEL_14:
          sub_6851C0(2890, a1 + 56);
          return;
        }
        goto LABEL_19;
      }
    }
    else
    {
      while ( v4 == 12 )
      {
        v3 = *(_QWORD *)(v3 + 160);
        v4 = *(_BYTE *)(v3 + 140);
      }
    }
    if ( v4 )
    {
      if ( (unsigned __int8)(v4 - 9) > 2u || (*(_BYTE *)(v3 + 177) & 0x10) == 0 )
        goto LABEL_14;
      v5 = *(_QWORD *)(a2 + 88);
      if ( *(_QWORD *)(v5 + 88) )
      {
        if ( (*(_BYTE *)(v5 + 160) & 1) == 0 )
          a2 = *(_QWORD *)(v5 + 88);
      }
      if ( a2 != **(_QWORD **)(*(_QWORD *)(v3 + 168) + 160LL) )
        goto LABEL_14;
    }
LABEL_19:
    v8 = *(_QWORD *)(v2 + 168);
    *(_BYTE *)(v8 + 21) |= 1u;
    *(_BYTE *)(v8 + 17) |= 1u;
    *(_QWORD *)(v8 + 40) = v3;
  }
}
