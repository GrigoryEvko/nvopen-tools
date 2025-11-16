// Function: sub_1453B30
// Address: 0x1453b30
//
void __fastcall sub_1453B30(char *src, char *a2)
{
  char *v4; // r12
  __int64 v5; // rbx
  __int16 v6; // cx
  __int16 v7; // dx
  int v8; // edi
  int v9; // esi
  char *i; // rax
  int v11; // edi
  int v12; // ecx
  __int64 v13; // rdx
  __int16 v14; // si

  if ( src != a2 )
  {
    v4 = src + 8;
LABEL_3:
    if ( a2 == v4 )
      return;
    while ( 1 )
    {
      v5 = *(_QWORD *)v4;
      v6 = *(_WORD *)(*(_QWORD *)v4 + 24LL);
      v7 = *(_WORD *)(*(_QWORD *)src + 24LL);
      if ( v6 == 5 )
      {
        v8 = *(_DWORD *)(v5 + 40);
        v9 = 1;
        if ( v7 != 5 )
          goto LABEL_7;
        goto LABEL_6;
      }
      v8 = 1;
      if ( v7 != 5 )
        break;
LABEL_6:
      v9 = *(_DWORD *)(*(_QWORD *)src + 40LL);
LABEL_7:
      if ( v9 >= v8 )
        break;
      if ( src != v4 )
        memmove(src + 8, src, v4 - src);
      v4 += 8;
      *(_QWORD *)src = v5;
      if ( a2 == v4 )
        return;
    }
    for ( i = v4; ; i -= 8 )
    {
      v13 = *((_QWORD *)i - 1);
      v14 = *(_WORD *)(v13 + 24);
      if ( v6 == 5 )
      {
        v11 = *(_DWORD *)(v5 + 40);
        v12 = 1;
        if ( v14 != 5 )
          goto LABEL_15;
      }
      else
      {
        if ( v14 != 5 )
        {
LABEL_21:
          *(_QWORD *)i = v5;
          v4 += 8;
          goto LABEL_3;
        }
        v11 = 1;
      }
      v12 = *(_DWORD *)(v13 + 40);
LABEL_15:
      if ( v12 >= v11 )
        goto LABEL_21;
      *(_QWORD *)i = v13;
      v6 = *(_WORD *)(v5 + 24);
    }
  }
}
