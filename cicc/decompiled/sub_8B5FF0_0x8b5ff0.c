// Function: sub_8B5FF0
// Address: 0x8b5ff0
//
void __fastcall sub_8B5FF0(_QWORD **a1, unsigned __int64 a2, __int64 *a3)
{
  unsigned __int64 v3; // r14
  char v4; // al
  int v5; // ebx
  _QWORD *v6; // r12
  _QWORD *v7; // r15
  int v8; // eax
  _QWORD *v9; // r13
  unsigned __int64 v10; // rsi
  char v11; // al
  _QWORD *v12; // rax
  __int64 *v13; // rdi
  _QWORD *v14; // rax

  v3 = a2;
  v4 = *(_BYTE *)(a2 + 80);
  if ( v4 == 16 )
  {
    v3 = **(_QWORD **)(a2 + 88);
    v4 = *(_BYTE *)(v3 + 80);
  }
  if ( v4 == 24 )
    v3 = *(_QWORD *)(v3 + 88);
  v5 = 0;
  v6 = 0;
  v7 = *a1;
  if ( !*a1 )
    goto LABEL_25;
  do
  {
    while ( 1 )
    {
      v9 = v7;
      v7 = (_QWORD *)*v7;
      v10 = v9[1];
      v11 = *(_BYTE *)(v10 + 80);
      if ( v11 == 16 )
      {
        v10 = **(_QWORD **)(v10 + 88);
        v11 = *(_BYTE *)(v10 + 80);
      }
      if ( v11 == 24 )
        v10 = *(_QWORD *)(v10 + 88);
      if ( ((*(_BYTE *)(v3 + 80) - 19) & 0xFD) != 0 )
        break;
      v8 = sub_8A6950(v3, v10);
      if ( v8 != 1 )
        goto LABEL_8;
LABEL_17:
      v12 = (_QWORD *)*v9;
      if ( v6 )
        *v6 = v12;
      else
        *a1 = v12;
      v13 = (__int64 *)v9[2];
      if ( v13 )
        sub_725130(v13);
      *v9 = qword_4F601A8;
      qword_4F601A8 = (__int64)v9;
      if ( !v7 )
        goto LABEL_22;
    }
    v8 = sub_8B2440(v3, v10, 1, 0);
    if ( v8 == 1 )
      goto LABEL_17;
LABEL_8:
    v6 = v9;
    if ( v8 == -1 )
      v5 = 1;
  }
  while ( v7 );
LABEL_22:
  if ( !v5 )
  {
LABEL_25:
    v14 = (_QWORD *)qword_4F601A8;
    if ( qword_4F601A8 )
      qword_4F601A8 = *(_QWORD *)qword_4F601A8;
    else
      v14 = (_QWORD *)sub_823970(24);
    *v14 = 0;
    v14[1] = a2;
    v14[2] = a3;
    *v14 = *a1;
    *a1 = v14;
    return;
  }
  if ( a3 )
    sub_725130(a3);
}
