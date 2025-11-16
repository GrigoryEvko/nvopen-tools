// Function: sub_5CB700
// Address: 0x5cb700
//
_QWORD *__fastcall sub_5CB700(__int64 a1, char a2, _DWORD *a3, _QWORD *a4)
{
  char v5; // al
  __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rdx
  char v9; // cl
  char v10; // cl
  _QWORD *result; // rax
  __int64 *v12; // r14
  _BYTE *v13; // r15
  __int64 **v14; // rax
  __int64 v15; // rdx
  char v18; // [rsp+2Eh] [rbp-32h] BYREF
  _BYTE v19[49]; // [rsp+2Fh] [rbp-31h] BYREF

  v5 = *(_BYTE *)(a1 + 8);
  v6 = qword_4CF7C60;
  v18 = v5;
  if ( !qword_4CF7C60 )
  {
    v12 = (__int64 *)&unk_4CF7AA0;
    v13 = &unk_4A428E0;
    qword_4CF7C60 = sub_881A70(0xFFFFFFFFLL, 28, 1, 2);
    v6 = qword_4CF7C60;
    do
    {
      v19[0] = *v13;
      v14 = (__int64 **)sub_881B20(v6, v19, 1);
      v6 = qword_4CF7C60;
      v15 = (__int64)*v14;
      v12[1] = (__int64)v13;
      v13 += 16;
      *v12 = v15;
      *v14 = v12;
      v12 += 2;
    }
    while ( v12 != &qword_4CF7C60 );
  }
  v7 = (_QWORD *)sub_881B20(v6, &v18, 0);
  if ( v7 )
  {
    while ( 1 )
    {
      v7 = (_QWORD *)*v7;
      if ( !v7 )
        break;
      v8 = v7[1];
      v9 = *(_BYTE *)(v8 + 1);
      if ( v9 == 6 || v9 == *(_BYTE *)(a1 + 9) )
      {
        v10 = *(_BYTE *)(v8 + 2);
        if ( v10 == 87 || v10 == a2 )
        {
          *a3 = *(_DWORD *)(v8 + 4);
          result = *(_QWORD **)(v7[1] + 8LL);
          *a4 = result;
          return result;
        }
      }
    }
  }
  *a3 = 0;
  *a4 = 0;
  return a4;
}
