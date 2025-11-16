// Function: sub_857250
// Address: 0x857250
//
__int64 __fastcall sub_857250(int a1, __int64 *a2)
{
  _QWORD *v2; // r13
  __int64 v3; // rbx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  const char *v18; // [rsp+18h] [rbp-98h]
  _BYTE v19[4]; // [rsp+2Ch] [rbp-84h] BYREF
  const char *v20; // [rsp+30h] [rbp-80h] BYREF
  unsigned __int64 v21; // [rsp+38h] [rbp-78h] BYREF
  _QWORD v22[2]; // [rsp+40h] [rbp-70h] BYREF
  int v23; // [rsp+50h] [rbp-60h]
  __int64 v24; // [rsp+58h] [rbp-58h]
  int v25; // [rsp+68h] [rbp-48h]
  char v26; // [rsp+6Ch] [rbp-44h]

  v2 = (_QWORD *)qword_4F5FCB0;
  v3 = ((1LL << (unk_4F06B9C - 1)) - 1) | (1LL << (unk_4F06B9C - 1));
  v20 = qword_4F06410 + 1;
  v18 = &qword_4F06410[qword_4F06400 - 1];
  sub_823800(qword_4F5FCB0);
  v22[1] = 0;
  v22[0] = &v20;
  v23 = 0;
  v24 = 0;
  v25 = 0x10000;
  v26 = 0;
  if ( v18 <= v20 )
  {
    v8 = v2[2];
  }
  else
  {
    do
    {
      sub_7CD070((__int64)v22, a1, (__int64 *)&v21, v3, 1, 0);
      if ( unk_4F064A8 || v21 <= 0x7F )
      {
        v11 = v2[2];
      }
      else
      {
        sub_722A20(v21, v19);
        v14 = v2[2];
        if ( (unsigned __int64)(v14 + 1) > v2[1] )
        {
          sub_823810(v2, v14 + 1, v12, v13, v6, v7);
          v14 = v2[2];
        }
        v10 = v19[0];
        *(_BYTE *)(v2[4] + v14) = v19[0];
        v9 = v19[1];
        v11 = v2[2] + 1LL;
        v2[2] = v11;
        v21 = v9;
      }
      if ( (unsigned __int64)(v11 + 1) > v2[1] )
      {
        sub_823810(v2, v11 + 1, v9, v10, v6, v7);
        v11 = v2[2];
      }
      v4 = v2[4];
      v5 = v21;
      *(_BYTE *)(v4 + v11) = v21;
      v8 = v2[2] + 1LL;
      v2[2] = v8;
    }
    while ( v20 < v18 );
  }
  if ( (unsigned __int64)(v8 + 1) > v2[1] )
  {
    sub_823810(v2, v8 + 1, v4, v5, v6, v7);
    v8 = v2[2];
  }
  *(_BYTE *)(v2[4] + v8) = 0;
  v15 = v2[4];
  v16 = v2[2] + 1LL;
  v2[2] = v16;
  *a2 = v16;
  return v15;
}
