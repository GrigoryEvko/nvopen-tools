// Function: sub_5C9A20
// Address: 0x5c9a20
//
__int64 __fastcall sub_5C9A20(__int64 a1, __int64 a2)
{
  char v4; // dl
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  int v8; // r15d
  char v9; // al
  __int64 v10; // r14
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 result; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // r15
  __int64 v18; // rdi
  _DWORD v19[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = *(_BYTE *)(a2 + 140);
  v5 = *(_QWORD *)(a1 + 32);
  if ( v4 == 12 )
  {
    v6 = a2;
    do
      v6 = *(_QWORD *)(v6 + 160);
    while ( *(_BYTE *)(v6 + 140) == 12 );
    v19[0] = 0;
    v7 = a2;
    do
    {
      v7 = *(_QWORD *)(v7 + 160);
      v4 = *(_BYTE *)(v7 + 140);
    }
    while ( v4 == 12 );
  }
  else
  {
    v19[0] = 0;
    v6 = a2;
  }
  v8 = 1;
  if ( v4 )
  {
    v9 = *(_BYTE *)(a1 + 8);
    if ( v9 == 67 )
    {
      v8 = 0;
      if ( (unsigned int)sub_8D2C40(v6) )
        goto LABEL_9;
      v15 = a1 + 56;
      v16 = 3409;
      goto LABEL_19;
    }
    v8 = 0;
    if ( v9 == 68 && !(unsigned int)sub_8D2CC0(v6) )
    {
      v15 = a1 + 56;
      v16 = 3410;
LABEL_19:
      v17 = sub_67D610(v16, v15, 8);
      sub_67F100(v17, a2);
      v18 = v17;
      v8 = 1;
      sub_685910(v18);
    }
  }
LABEL_9:
  v10 = *(_QWORD *)(v5 + 40);
  if ( *(_BYTE *)(v10 + 173) == 12 )
  {
    sub_6851C0(1689, a1 + 56);
  }
  else
  {
    v11 = *(_QWORD *)(v6 + 128);
    v12 = sub_620FD0(v10, v19);
    if ( v19[0] || ((v12 * v11 - 8) & 0xFFFFFFFFFFFFFFF7LL) != 0 )
    {
      sub_6851C0(3413, a1 + 56);
    }
    else if ( !v8 )
    {
      result = sub_7259C0(15);
      v14 = *(_QWORD *)(a1 + 56);
      *(_QWORD *)(result + 160) = a2;
      *(_QWORD *)(result + 64) = v14;
      *(_QWORD *)(result + 128) = v12 * v11;
      *(_QWORD *)(result + 168) = v10;
      *(_DWORD *)(result + 136) = v12 * v11;
      *(_BYTE *)(result + 177) = (*(_BYTE *)(a1 + 8) == 68) + 2;
      return result;
    }
  }
  *(_BYTE *)(a1 + 8) = 0;
  return sub_72C930();
}
