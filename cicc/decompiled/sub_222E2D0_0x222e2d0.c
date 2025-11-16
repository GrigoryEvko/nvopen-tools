// Function: sub_222E2D0
// Address: 0x222e2d0
//
void __fastcall sub_222E2D0(_BYTE *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  char v9; // r12
  __int64 v10; // rax
  _QWORD *v11; // r12
  _BYTE *v12; // rax
  __int64 v13; // r13
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int8 *v16; // rax

  v6 = *a2;
  *a1 = 0;
  v7 = (__int64)a2 + *(_QWORD *)(v6 - 24);
  v8 = *(unsigned int *)(v7 + 32);
  if ( (_DWORD)v8 )
    goto LABEL_6;
  v9 = a3;
  if ( !*(_QWORD *)(v7 + 216) )
  {
    if ( (_BYTE)a3 || (*(_BYTE *)(v7 + 25) & 0x10) == 0 )
      goto LABEL_9;
    goto LABEL_11;
  }
  sub_223DF30(*(_QWORD *)(v7 + 216), v8, a3);
  v10 = *a2;
  if ( v9 )
  {
LABEL_4:
    v7 = (__int64)a2 + *(_QWORD *)(v10 - 24);
    goto LABEL_5;
  }
  v7 = (__int64)a2 + *(_QWORD *)(v10 - 24);
  if ( (*(_BYTE *)(v7 + 25) & 0x10) != 0 )
  {
LABEL_11:
    v11 = *(_QWORD **)(v7 + 232);
    v12 = (_BYTE *)v11[2];
    if ( (unsigned __int64)v12 >= v11[3] )
    {
      LODWORD(v12) = (*(__int64 (__fastcall **)(_QWORD))(*v11 + 72LL))(*(_QWORD *)(v7 + 232));
      a3 = *a2;
      v7 = (__int64)a2 + *(_QWORD *)(*a2 - 24);
      v13 = *(_QWORD *)(v7 + 240);
      if ( v13 )
      {
        if ( (_DWORD)v12 == -1 )
        {
LABEL_21:
          LODWORD(v8) = *(_DWORD *)(v7 + 32) | 2;
          goto LABEL_6;
        }
        goto LABEL_13;
      }
    }
    else
    {
      v13 = *(_QWORD *)(v7 + 240);
      LOBYTE(v12) = *v12;
      if ( v13 )
      {
LABEL_13:
        while ( (*(_BYTE *)(*(_QWORD *)(v13 + 48) + 2LL * (unsigned __int8)v12 + 1) & 0x20) != 0 )
        {
          while ( 1 )
          {
            v14 = v11[2];
            v15 = v11[3];
            if ( v14 >= v15 )
            {
              if ( (*(unsigned int (__fastcall **)(_QWORD *))(*v11 + 80LL))(v11) == -1 )
                goto LABEL_20;
              v16 = (unsigned __int8 *)v11[2];
              v15 = v11[3];
            }
            else
            {
              v16 = (unsigned __int8 *)(v14 + 1);
              v11[2] = v16;
            }
            if ( (unsigned __int64)v16 >= v15 )
              break;
            if ( (*(_BYTE *)(*(_QWORD *)(v13 + 48) + 2LL * *v16 + 1) & 0x20) == 0 )
              goto LABEL_18;
          }
          LODWORD(v12) = (*(__int64 (__fastcall **)(_QWORD *))(*v11 + 72LL))(v11);
          if ( (_DWORD)v12 == -1 )
          {
LABEL_20:
            v7 = (__int64)a2 + *(_QWORD *)(*a2 - 24);
            goto LABEL_21;
          }
        }
LABEL_18:
        v10 = *a2;
        goto LABEL_4;
      }
    }
    sub_426219(v7, v8, a3, a4);
  }
LABEL_5:
  LODWORD(v8) = *(_DWORD *)(v7 + 32);
  if ( (_DWORD)v8 )
  {
LABEL_6:
    sub_222DC80(v7, v8 | 4);
    return;
  }
LABEL_9:
  *a1 = 1;
}
