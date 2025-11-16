// Function: sub_770120
// Address: 0x770120
//
__int64 __fastcall sub_770120(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r9d
  unsigned __int64 v4; // r13
  __int64 v5; // r8
  unsigned __int64 v8; // rsi
  __int64 v10; // rbx
  unsigned int i; // edx
  __int64 v12; // rax
  int v13; // eax
  unsigned __int64 *v14; // r15
  __int64 v15; // rax
  unsigned int v16; // [rsp+Ch] [rbp-44h]
  _QWORD v17[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = 0;
  v4 = *a1;
  v5 = *(_QWORD *)(*(_QWORD *)*a1 + 96LL);
  if ( *(_BYTE *)(v5 + 80) == 8 )
  {
    v8 = *(_QWORD *)(v5 + 88);
    v10 = *(_QWORD *)(v8 + 120);
    for ( v17[0] = v8; *(_BYTE *)(v10 + 140) == 12; v10 = *(_QWORD *)(v10 + 160) )
      ;
    for ( i = qword_4F08388 & (v8 >> 3); ; i = qword_4F08388 & (i + 1) )
    {
      v12 = qword_4F08380 + 16LL * i;
      if ( v8 == *(_QWORD *)v12 )
      {
        v3 = *(_DWORD *)(v12 + 8);
        goto LABEL_10;
      }
      if ( !*(_QWORD *)v12 )
        break;
    }
    v3 = 0;
LABEL_10:
    if ( *(_QWORD *)(v5 + 96) )
    {
      v16 = v3;
      v13 = sub_770120(v17, a2, a3, qword_4F08380);
      v8 = v17[0];
      v3 = v13 + v16;
    }
    v14 = (unsigned __int64 *)(v3 + a2);
    v15 = -(((unsigned int)((_DWORD)v14 - a3) >> 3) + 10);
    *(_BYTE *)(a3 + v15) |= 1 << (((_BYTE)v14 - a3) & 7);
    if ( *(_BYTE *)(v10 + 140) == 11 )
      *v14 = v4;
    else
      *v14 = 0;
    *a1 = v8;
  }
  return v3;
}
