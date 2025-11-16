// Function: sub_813620
// Address: 0x813620
//
void __fastcall sub_813620(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  char v4; // al
  char *v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD v9[3]; // [rsp+18h] [rbp-18h] BYREF

  v2 = *(_QWORD *)(a1 + 168);
  if ( (unsigned __int8)(*(_BYTE *)(a1 + 140) - 9) > 2u )
  {
LABEL_9:
    v6 = *(_QWORD *)(v2 + 168);
    sub_810560(a1, a2);
    if ( v6 )
    {
      v9[0] = v6;
      sub_811CB0(v9, 0, 0, a2);
    }
    return;
  }
  v3 = *(_QWORD *)(v2 + 256);
  if ( !v3 )
  {
    if ( (*(_BYTE *)(a1 + 177) & 0x10) != 0 )
    {
      v7 = sub_880FA0(a1);
      if ( v7 )
      {
        v8 = *(_QWORD *)(v7 + 88);
        v2 = *(_QWORD *)(a1 + 168);
        if ( (*(_BYTE *)(v8 + 266) & 1) != 0 )
        {
          sub_812B60((unsigned int *)(*(_QWORD *)(v8 + 104) + 128LL), *(_QWORD *)(v2 + 168), a2);
          return;
        }
      }
      else
      {
        v2 = *(_QWORD *)(a1 + 168);
      }
    }
    goto LABEL_9;
  }
  if ( *(_BYTE *)(v3 + 140) == 14 )
  {
    v4 = *(_BYTE *)(v3 + 160);
    if ( v4 == 1 )
    {
      if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
        v5 = *(char **)(a1 + 24);
      else
        v5 = *(char **)(a1 + 8);
      if ( !v5 )
        v5 = "?";
      goto LABEL_7;
    }
    if ( v4 == 2 )
    {
      v5 = "?";
LABEL_7:
      sub_80BC40(v5, a2);
      return;
    }
    if ( v4 )
      sub_721090();
    sub_812B60((unsigned int *)(*(_QWORD *)(v3 + 168) + 24LL), 0, a2);
  }
  else
  {
    sub_80F5E0(v3, 1, a2);
  }
}
