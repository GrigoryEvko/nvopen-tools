// Function: sub_5E8530
// Address: 0x5e8530
//
__int64 __fastcall sub_5E8530(_QWORD *a1, unsigned int a2)
{
  _QWORD *v2; // rax
  __int64 result; // rax
  _QWORD *v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // rbx
  __int64 v13; // rdi
  _QWORD *v14; // rbx
  _QWORD *v15; // [rsp+0h] [rbp-240h]
  __int64 v16; // [rsp+18h] [rbp-228h]
  unsigned int v17; // [rsp+2Ch] [rbp-214h] BYREF
  _QWORD v18[59]; // [rsp+30h] [rbp-210h] BYREF
  __int64 v19; // [rsp+208h] [rbp-38h] BYREF

  v15 = a1;
  if ( *(char *)(a1[21] + 110LL) < 0 )
  {
    v2 = a1;
    do
      v2 = *(_QWORD **)(v2[5] + 32LL);
    while ( *(char *)(v2[21] + 110LL) < 0 );
    v15 = v2;
  }
  result = *(_QWORD *)(*v15 + 96LL);
  v4 = *(_QWORD **)(result + 64);
  v16 = result;
  if ( v4 )
  {
    while ( 1 )
    {
      v5 = *(_QWORD *)(v4[1] + 64LL);
      *(_QWORD *)(v16 + 64) = *v4;
      sub_7B8190();
      v6 = unk_4F04C68 + 776LL * unk_4F04C64;
      if ( *(_BYTE *)(v6 + 4) != 6
        || (v7 = *(_QWORD *)(v6 + 208), v8 = 0, v5 != v7)
        && (!v5 || !v7 || !dword_4F07588 || (v9 = *(_QWORD *)(v7 + 32), *(_QWORD *)(v5 + 32) != v9) || !v9) )
      {
        v8 = 1;
        sub_865D70(v5, a2, 0, 1, 1, 0);
      }
      sub_7296C0(&v17);
      sub_7BC160(v4[2]);
      memset(v18, 0, sizeof(v18));
      v18[19] = v18;
      v18[3] = *(_QWORD *)&dword_4F063F8;
      if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
        BYTE2(v18[22]) |= 1u;
      v10 = v4[1];
      v11 = *(_BYTE *)(v10 + 80) == 8;
      v18[0] = v10;
      if ( !v11 )
        sub_721090(&v19);
      v12 = *(_QWORD *)(v10 + 104);
      sub_63B0A0(v18);
      if ( *(_QWORD *)(v12 + 8) )
        v4[2] = 0;
      if ( word_4F06418[0] != 9 )
      {
        sub_6851C0(65, &dword_4F063F8);
        while ( word_4F06418[0] != 9 )
          sub_7B8B50();
      }
      sub_7B8B50();
      sub_729730(v17);
      if ( v8 )
        sub_866010();
      v13 = v4[2];
      v14 = *(_QWORD **)(v16 + 64);
      if ( v13 )
        sub_7AEA70(v13);
      *v4 = qword_4CF7FE0;
      qword_4CF7FE0 = (__int64)v4;
      sub_7B8260();
      if ( !v14 )
        break;
      v4 = v14;
    }
    result = *(int *)(*(_QWORD *)(v15[21] + 152LL) + 240LL);
    if ( (_DWORD)result != -1 )
    {
      result = unk_4F04C68 + 776 * result;
      *(_QWORD *)(result + 272) = 0;
    }
  }
  return result;
}
