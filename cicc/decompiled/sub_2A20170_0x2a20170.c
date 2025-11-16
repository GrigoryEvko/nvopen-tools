// Function: sub_2A20170
// Address: 0x2a20170
//
char __fastcall sub_2A20170(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  char result; // al
  char v16; // [rsp+Fh] [rbp-51h]
  _BYTE *v17; // [rsp+10h] [rbp-50h] BYREF
  __int64 v18; // [rsp+18h] [rbp-48h]
  _BYTE v19[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_17;
  v11 = v5[7];
  if ( !v11 )
    goto LABEL_17;
  v12 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_500A7C8 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_15:
  if ( v12 == v5 + 6 || dword_500A7C8 < *((_DWORD *)v12 + 8) || (result = qword_500A848, !*((_DWORD *)v12 + 9)) )
  {
LABEL_17:
    v18 = 0x400000000LL;
    v17 = v19;
    sub_D46D90(a1, (__int64)&v17);
    result = 0;
    if ( (unsigned int)v18 <= 2 )
    {
      result = 1;
      if ( *(_DWORD *)(a2 + 8) )
      {
        result = 0;
        if ( *(_DWORD *)(a2 + 8) == 1 )
        {
          result = qword_500A768;
          if ( !(_BYTE)qword_500A768 )
            result = sub_AA5820(**(_QWORD **)a2, (__int64)&v17) != 0;
        }
      }
    }
    if ( v17 != v19 )
    {
      v16 = result;
      _libc_free((unsigned __int64)v17);
      return v16;
    }
  }
  return result;
}
