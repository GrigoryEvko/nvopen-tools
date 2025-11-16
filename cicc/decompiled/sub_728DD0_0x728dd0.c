// Function: sub_728DD0
// Address: 0x728dd0
//
void __fastcall sub_728DD0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  _QWORD *v3; // rcx
  __int64 v4; // r8
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  _BOOL4 v7; // r9d
  _QWORD *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  bool v13; // zf

  if ( *(_BYTE *)(a1 + 24) == 17 )
  {
    v1 = *(_QWORD *)(a1 + 56);
    v2 = *(_QWORD *)(v1 + 56);
    if ( v2 )
    {
      v3 = *(_QWORD **)(*(_QWORD *)(v1 + 80) + 8LL);
      if ( !v3 )
        goto LABEL_14;
      v4 = v3[2];
      if ( !v4 )
        goto LABEL_14;
      v5 = *(_QWORD **)(v4 + 160);
      if ( v5 )
      {
        v6 = (_QWORD *)(v4 + 160);
        v7 = 0;
      }
      else
      {
        v11 = *(int *)(v4 + 240);
        if ( (_DWORD)v11 == -1 )
        {
LABEL_14:
          v10 = sub_86A500(v2);
          sub_86A190(v2, v10);
          return;
        }
        v12 = qword_4F04C68[0] + 776 * v11;
        v13 = *(_QWORD *)(v12 + 312) == (_QWORD)v3;
        v6 = (_QWORD *)(v12 + 304);
        v5 = *(_QWORD **)(v12 + 304);
        v7 = v13;
        if ( !v5 )
        {
LABEL_12:
          if ( v7 )
            *(_QWORD *)(qword_4F04C68[0] + 776LL * *(int *)(v4 + 240) + 312) = v5;
          goto LABEL_14;
        }
      }
      v8 = 0;
      while ( 1 )
      {
        v9 = *v5;
        if ( v3 == v5 )
          break;
        v6 = v5;
        v8 = v5;
        if ( !v9 )
          goto LABEL_12;
        v5 = (_QWORD *)*v5;
      }
      *v6 = v9;
      v5 = v8;
      goto LABEL_12;
    }
  }
}
