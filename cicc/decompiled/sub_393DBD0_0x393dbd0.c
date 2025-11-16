// Function: sub_393DBD0
// Address: 0x393dbd0
//
__int64 __fastcall sub_393DBD0(_QWORD *a1, _QWORD *a2, _QWORD *a3, _QWORD *a4)
{
  unsigned int v4; // r15d
  _BYTE *v7; // rcx
  unsigned __int64 v8; // r8
  __int64 i; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  unsigned int v19; // eax
  unsigned __int64 v20; // r13
  __int64 v21; // rsi
  unsigned __int64 v22; // rdi
  unsigned __int64 v24[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = 0;
  v7 = (_BYTE *)*a1;
  if ( *(_BYTE *)*a1 != 32 )
  {
    v8 = a1[1];
    for ( i = a1[1]; ; --i )
    {
      if ( !i )
      {
        v13 = -2;
        v12 = -1;
        goto LABEL_7;
      }
      v12 = i - 1;
      if ( v7[i - 1] == 58 )
        break;
    }
    v13 = i - 2;
LABEL_7:
    v14 = v13;
    if ( v8 <= v13 )
      v14 = a1[1];
    while ( 1 )
    {
      if ( !v14 )
      {
        v16 = v12;
        goto LABEL_14;
      }
      v15 = v14 - 1;
      if ( v7[v14 - 1] == 58 )
        break;
      --v14;
    }
    v16 = v13 - v15;
    if ( v8 > v15 )
      v8 = v14 - 1;
LABEL_14:
    *a2 = v7;
    a2[1] = v8;
    v17 = a1[1];
    if ( v14 > v17 )
      v14 = a1[1];
    v18 = v17 - v14;
    if ( v18 > v16 )
      v18 = v16;
    LOBYTE(v19) = sub_16D2B80(v14 + *a1, v18, 0xAu, v24);
    v4 = v19;
    if ( (_BYTE)v19 )
    {
      return 0;
    }
    else
    {
      v20 = v12 + 1;
      v21 = 0;
      *a3 = v24[0];
      v22 = a1[1];
      if ( v20 <= v22 )
      {
        v21 = v22 - v20;
        v22 = v20;
      }
      if ( !sub_16D2B80(*a1 + v22, v21, 0xAu, v24) )
      {
        v4 = 1;
        *a4 = v24[0];
      }
    }
  }
  return v4;
}
