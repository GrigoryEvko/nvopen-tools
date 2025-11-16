// Function: sub_24A2790
// Address: 0x24a2790
//
_QWORD *__fastcall sub_24A2790(_QWORD *a1, int *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rsi
  _QWORD *v11; // rdx
  char v12; // r15
  __int64 v13; // r13
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-50h]
  __int64 v17; // [rsp+10h] [rbp-40h]
  __int64 v18; // [rsp+18h] [rbp-38h]

  if ( !*a2 )
  {
    a1[6] = 0;
    a1[1] = a1 + 4;
    a1[7] = a1 + 10;
    goto LABEL_3;
  }
  v16 = sub_BC0510(a4, &unk_4F87C68, a3);
  v8 = sub_BC0510(a4, &unk_4F82418, a3);
  v9 = *(_QWORD *)(a3 + 32);
  v10 = a1 + 4;
  v11 = a1 + 10;
  v17 = *(_QWORD *)(v8 + 8);
  v18 = a3 + 24;
  if ( v9 == a3 + 24 )
    goto LABEL_30;
  v12 = 0;
  do
  {
    v13 = 0;
    if ( v9 )
      v13 = v9 - 56;
    if ( !sub_B2FC80(v13)
      && !(unsigned __int8)sub_B2D610(v13, 48)
      && !(unsigned __int8)sub_B2D610(v13, 47)
      && !(unsigned __int8)sub_B2D610(v13, 18)
      && !(unsigned __int8)sub_B2D610(v13, 18) )
    {
      if ( (unsigned __int8)sub_B2D610(v13, 5)
        || *(_QWORD *)(v16 + 16)
        && (v14 = sub_BC1CD0(v17, &unk_4F8D9A8, v13), sub_24A25C0(v16 + 8, v13, (__int64 *)(v14 + 8))) )
      {
        v15 = *a2;
        if ( *a2 == 2 )
        {
          v12 = 1;
          sub_B2CD30(v13, 18);
        }
        else
        {
          if ( v15 > 2 )
          {
            if ( v15 == 3 )
            {
              if ( (unsigned __int8)sub_B2D610(v13, 3) )
                goto LABEL_12;
              sub_B2CD30(v13, 48);
              sub_B2CD30(v13, 31);
            }
          }
          else
          {
            if ( !v15 )
              BUG();
            if ( v15 == 1 )
            {
              v12 = 1;
              sub_B2CD30(v13, 47);
              goto LABEL_12;
            }
          }
          v12 = 1;
        }
      }
    }
LABEL_12:
    v9 = *(_QWORD *)(v9 + 8);
  }
  while ( v18 != v9 );
  v10 = a1 + 4;
  v11 = a1 + 10;
  if ( v12 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v10;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v11;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
    return a1;
  }
LABEL_30:
  a1[1] = v10;
  a1[6] = 0;
  a1[7] = v11;
LABEL_3:
  *((_BYTE *)a1 + 76) = 1;
  a1[2] = 0x100000002LL;
  a1[8] = 2;
  *((_DWORD *)a1 + 18) = 0;
  *((_DWORD *)a1 + 6) = 0;
  *((_BYTE *)a1 + 28) = 1;
  a1[4] = &qword_4F82400;
  *a1 = 1;
  return a1;
}
