// Function: sub_799890
// Address: 0x799890
//
__int64 __fastcall sub_799890(__int64 a1)
{
  _QWORD *v1; // rax
  int v2; // r12d
  __int64 v3; // rdi
  _QWORD *v4; // r15
  __int64 v5; // r8
  __int64 *v6; // r13
  unsigned int v7; // eax
  FILE *v8; // r14
  int v9; // r9d
  _QWORD *v10; // r12
  int v11; // r15d
  __int64 v12; // rbx
  __int64 result; // rax
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  char i; // al
  unsigned __int8 v17; // dl
  _QWORD *v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  int v22; // [rsp+24h] [rbp-4Ch]
  int v23; // [rsp+28h] [rbp-48h]
  unsigned int v24[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v1 = *(_QWORD **)(a1 + 48);
  v24[0] = 1;
  v18 = v1;
  while ( 1 )
  {
    v2 = 0;
    v22 = 1;
    v3 = v18[2];
    v4 = (_QWORD *)v18[3];
    v5 = v18[4];
    v6 = *(__int64 **)(v18[1] + 16LL);
    v7 = 0;
    v8 = (FILE *)v18[5];
    if ( *(_BYTE *)(v3 + 140) != 8 )
      break;
    v20 = v18[4];
    v22 = sub_8D4490(v3);
    v14 = sub_8D40F0(v3);
    v5 = v20;
    v15 = v14;
    for ( i = *(_BYTE *)(v14 + 140); i == 12; i = *(_BYTE *)(v15 + 140) )
      v15 = *(_QWORD *)(v15 + 160);
    v17 = i - 2;
    v7 = 16;
    if ( v17 > 1u )
    {
      v7 = sub_7764B0(a1, v15, v24);
      v5 = v20;
    }
    v2 = 1;
    if ( v22 > 0 )
      break;
LABEL_12:
    v18 = (_QWORD *)*v18;
    if ( !v18 )
      return v24[0];
  }
  v9 = v2;
  v10 = v4;
  v19 = v7;
  v11 = 0;
  v12 = v5;
  while ( 1 )
  {
    v23 = v9;
    result = sub_798FD0(a1, v6, v8, v10, v12, v9);
    if ( !(_DWORD)result )
      return result;
    ++v11;
    v10 = (_QWORD *)((char *)v10 + v19);
    v9 = v23;
    if ( v22 <= v11 )
      goto LABEL_12;
  }
}
