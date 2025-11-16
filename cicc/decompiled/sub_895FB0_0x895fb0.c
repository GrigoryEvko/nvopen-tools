// Function: sub_895FB0
// Address: 0x895fb0
//
void sub_895FB0()
{
  __int64 v0; // rbx
  __int64 v1; // rdx
  char v2; // al
  __int64 v3; // rcx
  int v4; // esi
  __int64 v5; // rdi
  __int64 *v6; // rax
  __int64 v7; // rdi
  _QWORD *v8; // rdi

  v0 = qword_4F601F0;
  if ( qword_4F601F0 )
  {
    while ( 1 )
    {
      v1 = *(_QWORD *)(v0 + 16);
      if ( (*(_BYTE *)(v1 + 28) & 3) != 1 )
        goto LABEL_3;
      v2 = *(_BYTE *)(v0 + 80);
      if ( (v2 & 8) != 0 )
        goto LABEL_3;
      v3 = *(_QWORD *)(v0 + 24);
      v4 = *(_DWORD *)(v1 + 24);
      if ( ((*(_BYTE *)(v3 + 80) - 7) & 0xFD) != 0 )
      {
        if ( v4 )
        {
          if ( (v2 & 0x20) != 0 )
          {
            v7 = *(_QWORD *)(v3 + 88);
            if ( *(_DWORD *)(v7 + 160) )
            {
LABEL_15:
              v8 = (_QWORD *)sub_72B800(v7);
              if ( v8 )
                sub_734690(v8);
            }
          }
        }
        else
        {
          v7 = *(_QWORD *)(v3 + 88);
          if ( *(_DWORD *)(v7 + 160) )
            goto LABEL_15;
        }
LABEL_3:
        v0 = *(_QWORD *)(v0 + 8);
        if ( !v0 )
          return;
      }
      else
      {
        if ( v4 && unk_4D04734 == 2 && (v2 & 0x20) == 0 )
          goto LABEL_3;
        v5 = *(_QWORD *)(v3 + 88);
        v6 = *(__int64 **)(v5 + 32);
        if ( v6 )
          v5 = *v6;
        if ( *(_BYTE *)(v5 + 136) )
          goto LABEL_3;
        sub_735A70(v5);
        v0 = *(_QWORD *)(v0 + 8);
        if ( !v0 )
          return;
      }
    }
  }
}
