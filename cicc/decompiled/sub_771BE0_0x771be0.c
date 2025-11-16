// Function: sub_771BE0
// Address: 0x771be0
//
__int64 sub_771BE0()
{
  __int64 *v0; // rax
  __int64 v1; // rdx
  void *v2; // rdi
  __int64 result; // rax
  __int64 v4; // rbx
  _OWORD *v5; // r14
  unsigned __int8 v6; // di
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // rax
  _DWORD v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  v0 = (__int64 *)qword_4F082A0;
  qword_4F083A8 = 0;
  if ( qword_4F082A0 )
  {
    qword_4F082A0 = *(_QWORD *)(qword_4F082A0 + 8);
    v1 = 0;
  }
  else
  {
    v0 = (__int64 *)sub_823970(0x10000);
    v1 = qword_4F083A8;
  }
  *v0 = v1;
  qword_4F083A8 = (__int64)v0;
  v0[1] = 0;
  v2 = qword_4F08370;
  qword_4F083B0 = 0;
  qword_4F083C0 = 0;
  qword_4F083A0 = qword_4F083A8 + 24;
  dword_4F083B8 = 1;
  if ( qword_4F08370 )
  {
    qword_4F08380 = (__int64)qword_4F08370;
    qword_4F08370 = *(void **)qword_4F08370;
  }
  else
  {
    qword_4F08380 = sub_823970(0x4000);
    v2 = (void *)qword_4F08380;
  }
  result = (__int64)memset(v2, 0, 0x4000u);
  qword_4F08388 = 1023;
  qword_4F082A8 = 0;
  qword_4F08098 = 0;
  qword_4F08090 = 0;
  qword_4F08088 = 0;
  qword_4F08080 = 0;
  qword_4F08068 = 0;
  qword_4F08060 = 0;
  if ( !dword_4F080A0 )
  {
    v4 = 0;
    sub_620D80(&xmmword_4F08290, 0);
    v5 = &unk_4F081A0;
    sub_620D80(&xmmword_4F08280, 1);
    do
    {
      sub_70B680(v4, 0, v5, v10);
      v6 = v4;
      v7 = 16 * v4++;
      ++v5;
      sub_70B680(v6, 1, (char *)&unk_4F080C0 + v7, v10);
    }
    while ( v4 != 14 );
    v8 = (_QWORD *)sub_72CBE0();
    v9 = sub_72D2E0(v8);
    dword_4F080A0 = 1;
    qword_4F080A8 = v9;
    result = dword_4D041A8;
    if ( dword_4D041A8 )
    {
      result = (__int64)sub_724D80(2);
      qword_4F08070 = result;
    }
  }
  return result;
}
