// Function: sub_1420670
// Address: 0x1420670
//
void __fastcall sub_1420670(__int64 a1, __int64 a2, char a3, char *a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // eax
  char v8; // al
  char v9; // dl
  int v10; // eax
  bool v11; // zf
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax

  if ( a3 )
  {
    if ( *(_BYTE *)(a1 + 16) == 22 )
    {
      v5 = *(_QWORD *)(a1 + 112);
      if ( a2 != v5 )
      {
        if ( v5 != 0 && v5 != -8 && v5 != -16 )
          sub_1649B30(a1 + 96);
        *(_QWORD *)(a1 + 112) = a2;
        if ( a2 != -8 && a2 != 0 && a2 != -16 )
          sub_164C220(a1 + 96);
      }
      v6 = *(_QWORD *)(a1 - 24);
      if ( *(_BYTE *)(v6 + 16) == 22 )
        v7 = *(_DWORD *)(v6 + 84);
      else
        v7 = *(_DWORD *)(v6 + 72);
      *(_DWORD *)(a1 + 88) = v7;
      v8 = *(_BYTE *)(a1 + 81);
      if ( a4[1] )
      {
LABEL_13:
        v9 = *a4;
        if ( v8 )
        {
          *(_BYTE *)(a1 + 80) = v9;
        }
        else
        {
          *(_BYTE *)(a1 + 80) = v9;
          *(_BYTE *)(a1 + 81) = 1;
        }
        return;
      }
    }
    else
    {
      if ( *(_BYTE *)(a2 + 16) == 22 )
        v10 = *(_DWORD *)(a2 + 84);
      else
        v10 = *(_DWORD *)(a2 + 72);
      v11 = *(_QWORD *)(a1 - 24) == 0;
      *(_DWORD *)(a1 + 84) = v10;
      if ( !v11 )
      {
        v12 = *(_QWORD *)(a1 - 16);
        v13 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *(_QWORD *)(a1 - 24) = a2;
      v14 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 - 16) = v14;
      if ( v14 )
        *(_QWORD *)(v14 + 16) = (a1 - 16) | *(_QWORD *)(v14 + 16) & 3LL;
      *(_QWORD *)(a1 - 8) = *(_QWORD *)(a1 - 8) & 3LL | (a2 + 8);
      *(_QWORD *)(a2 + 8) = a1 - 24;
      v8 = *(_BYTE *)(a1 + 81);
      if ( a4[1] )
        goto LABEL_13;
    }
    if ( v8 )
      *(_BYTE *)(a1 + 81) = 0;
  }
  else
  {
    if ( *(_QWORD *)(a1 - 24) )
    {
      v15 = *(_QWORD *)(a1 - 16);
      v16 = *(_QWORD *)(a1 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
    }
    *(_QWORD *)(a1 - 24) = a2;
    if ( a2 )
    {
      v17 = *(_QWORD *)(a2 + 8);
      *(_QWORD *)(a1 - 16) = v17;
      if ( v17 )
        *(_QWORD *)(v17 + 16) = (a1 - 16) | *(_QWORD *)(v17 + 16) & 3LL;
      *(_QWORD *)(a1 - 24 + 16) = (a2 + 8) | *(_QWORD *)(a1 - 8) & 3LL;
      *(_QWORD *)(a2 + 8) = a1 - 24;
    }
  }
}
