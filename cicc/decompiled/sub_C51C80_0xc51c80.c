// Function: sub_C51C80
// Address: 0xc51c80
//
void __fastcall sub_C51C80(__int64 a1)
{
  __int64 v1; // rdi
  _QWORD *v2; // rsi
  __int64 v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // r14
  unsigned __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v9; // r15
  void (__fastcall *v10)(__int64, _QWORD *, _QWORD); // r13
  _BYTE *v11; // [rsp+10h] [rbp-840h] BYREF
  __int64 v12; // [rsp+18h] [rbp-838h]
  _BYTE v13[2096]; // [rsp+20h] [rbp-830h] BYREF

  if ( !qword_4F83C80 )
    sub_C7D570(&qword_4F83C80, sub_C58C10, sub_C51550);
  if ( !*(_BYTE *)(qword_4F83C80 + 1176) )
  {
    if ( qword_4F83C80 )
    {
      if ( !*(_BYTE *)(qword_4F83C80 + 1376) )
        return;
    }
    else
    {
      sub_C7D570(&qword_4F83C80, sub_C58C10, sub_C51550);
      if ( !*(_BYTE *)(qword_4F83C80 + 1376) )
        return;
    }
  }
  v1 = *(_QWORD *)(a1 + 344);
  v2 = &v11;
  v11 = v13;
  v12 = 0x8000000000LL;
  sub_C50450((__int64 **)(v1 + 128), (__int64)&v11, 1);
  v3 = (unsigned int)v12;
  if ( (_DWORD)v12 )
  {
    v4 = 0;
    v5 = 0;
    do
    {
      v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)&v11[16 * v5 + 8] + 40LL))(*(_QWORD *)&v11[16 * v5 + 8]);
      if ( (unsigned __int64)v4 < v6 )
        v4 = (_QWORD *)v6;
      ++v5;
    }
    while ( v3 != v5 );
    v7 = (unsigned int)v12;
    if ( (_DWORD)v12 )
    {
      v8 = 0;
      do
      {
        v9 = *(_QWORD *)&v11[16 * v8 + 8];
        v10 = *(void (__fastcall **)(__int64, _QWORD *, _QWORD))(*(_QWORD *)v9 + 56LL);
        if ( !qword_4F83C80 )
          sub_C7D570(&qword_4F83C80, sub_C58C10, sub_C51550);
        ++v8;
        v2 = v4;
        v10(v9, v4, *(unsigned __int8 *)(qword_4F83C80 + 1376));
      }
      while ( v8 != v7 );
    }
  }
  if ( v11 != v13 )
    _libc_free(v11, v2);
}
