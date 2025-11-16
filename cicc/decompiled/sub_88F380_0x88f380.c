// Function: sub_88F380
// Address: 0x88f380
//
void __fastcall sub_88F380(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  int v4; // ecx
  char v5; // r12

  if ( a2 )
  {
    v2 = *(_QWORD *)(a2 + 104);
    if ( (*(_BYTE *)(v2 + 121) & 1) == 0 )
    {
      v3 = *(_QWORD *)(a1 + 240);
      v4 = *(_DWORD *)(a1 + 60);
      if ( v3 && *(char *)(v3 + 177) < 0 )
      {
        v5 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3 + 168) + 160LL) + 121LL) & 1 | (v4 != 0);
        if ( !v5 )
          goto LABEL_6;
      }
      else
      {
        v5 = 0;
        if ( !v4 )
        {
LABEL_6:
          *(_BYTE *)(v2 + 121) = v5 | *(_BYTE *)(v2 + 121) & 0xFE;
          return;
        }
      }
      v5 = 1;
      if ( *(_QWORD *)(a2 + 8) )
      {
        if ( !*(_DWORD *)(a1 + 36) )
        {
          sub_6851C0(0x42Fu, (_DWORD *)(a1 + 140));
          v2 = *(_QWORD *)(a2 + 104);
        }
      }
      goto LABEL_6;
    }
  }
}
