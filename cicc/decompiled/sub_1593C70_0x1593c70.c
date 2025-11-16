// Function: sub_1593C70
// Address: 0x1593c70
//
char __fastcall sub_1593C70(__int64 a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  unsigned int v5; // eax
  _QWORD *v6; // r15
  __int64 v7; // rbx
  __int64 i; // r13
  char v9; // dl
  __int64 v10; // r12
  _QWORD *v11; // rax
  unsigned int v12; // edi
  char result; // al
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  __int64 v16; // rdi

  if ( *(_BYTE *)(a1 + 16) != 5 )
    return 0;
  v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v5 )
  {
    v6 = a2;
    v7 = v5 - 1;
    for ( i = 0; ; ++i )
    {
      v10 = *(_QWORD *)(a1 + 24 * (i - v5));
      if ( *(_BYTE *)(v10 + 16) != 5 )
        goto LABEL_5;
      v11 = (_QWORD *)v6[1];
      if ( (_QWORD *)v6[2] != v11 )
        goto LABEL_4;
      a2 = &v11[*((unsigned int *)v6 + 7)];
      v12 = *((_DWORD *)v6 + 7);
      if ( v11 != a2 )
      {
        a4 = 0;
        while ( v10 != *v11 )
        {
          if ( *v11 == -2 )
            a4 = v11;
          if ( a2 == ++v11 )
          {
            if ( !a4 )
              goto LABEL_26;
            *a4 = v10;
            --*((_DWORD *)v6 + 8);
            ++*v6;
            goto LABEL_17;
          }
        }
        goto LABEL_5;
      }
LABEL_26:
      if ( v12 < *((_DWORD *)v6 + 6) )
      {
        *((_DWORD *)v6 + 7) = v12 + 1;
        *a2 = v10;
        ++*v6;
      }
      else
      {
LABEL_4:
        a2 = (_QWORD *)v10;
        sub_16CCBA0(v6, v10);
        if ( !v9 )
          goto LABEL_5;
      }
LABEL_17:
      a2 = v6;
      result = sub_1593C70(v10, v6);
      if ( result )
        return result;
LABEL_5:
      if ( i == v7 )
        break;
      v5 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
  }
  v14 = *(_WORD *)(a1 + 18);
  if ( v14 > 0x12u )
  {
    if ( (unsigned __int16)(v14 - 20) > 1u )
      return 0;
  }
  else if ( v14 <= 0x10u )
  {
    return 0;
  }
  v15 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v16 = *(_QWORD *)(a1 + 24 * (1 - v15));
  result = 1;
  if ( *(_BYTE *)(v16 + 16) == 13 )
    return sub_1593BB0(v16, (__int64)a2, v15, (__int64)a4);
  return result;
}
