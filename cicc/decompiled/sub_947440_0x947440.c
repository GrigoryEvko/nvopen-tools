// Function: sub_947440
// Address: 0x947440
//
__int64 __fastcall sub_947440(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        char a7,
        unsigned __int64 a8)
{
  unsigned __int64 v8; // rax
  int v11; // ebx
  char v12; // dl
  unsigned __int64 v13; // r10
  int v14; // eax
  char v15; // r15
  __int64 v16; // rdi
  int v17; // r14d
  __int64 v18; // rax
  int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // r15
  unsigned int *v22; // r14
  unsigned int *v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rsi
  char v27; // al
  int v28; // eax
  __int64 v29; // rax
  char v30; // al
  bool v31; // al
  __int64 v32; // [rsp+10h] [rbp-B0h]
  unsigned __int64 v33; // [rsp+10h] [rbp-B0h]
  unsigned __int8 v34; // [rsp+18h] [rbp-A8h]
  int v35; // [rsp+18h] [rbp-A8h]
  char v36; // [rsp+1Ch] [rbp-A4h]
  _BYTE v39[32]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v40; // [rsp+50h] [rbp-70h]
  char v41; // [rsp+60h] [rbp-60h] BYREF
  __int16 v42; // [rsp+80h] [rbp-40h]

  v8 = a3;
  v11 = 0;
  v36 = a4;
  v12 = a7;
  v13 = a8;
  if ( v8 )
  {
    _BitScanReverse64(&v8, v8);
    v14 = v8 ^ 0x3F;
    a4 = (unsigned int)(63 - v14);
    LOBYTE(v11) = 63 - v14;
    BYTE1(v11) = 1;
  }
  v15 = 0;
  if ( (_DWORD)a6 )
  {
    _BitScanReverse64((unsigned __int64 *)&a6, (unsigned int)a6);
    v15 = 1;
    v34 = 63 - (a6 ^ 0x3F);
  }
  v16 = *(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(v16 + 336) )
    goto LABEL_6;
  v27 = sub_918D80(v16 + 8, a8);
  v13 = a8;
  v12 = a7;
  if ( !v27 )
  {
    v16 = *(_QWORD *)(a1 + 32);
LABEL_6:
    v17 = 1;
    v40 = 257;
    if ( !v12 )
    {
      v17 = unk_4D0463C;
      if ( unk_4D0463C )
      {
        v33 = v13;
        v31 = sub_90AA40(v16, a5);
        v16 = *(_QWORD *)(a1 + 32);
        v13 = v33;
        v17 = v31;
      }
    }
    v18 = sub_91A390(v16 + 8, v13, 0, a4);
    v19 = v34;
    v32 = v18;
    if ( !v15 )
    {
      v29 = sub_AA4E30(*(_QWORD *)(a1 + 96));
      v30 = sub_AE5020(v29, v32);
      v19 = v34;
      LOBYTE(v19) = v30;
    }
    v35 = v19;
    v42 = 257;
    v20 = sub_BD2C40(80, unk_3F10A14);
    v21 = v20;
    if ( v20 )
      sub_B4D190(v20, v32, a5, (unsigned int)&v41, v17, v35, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 136) + 16LL))(
      *(_QWORD *)(a1 + 136),
      v21,
      v39,
      *(_QWORD *)(a1 + 104),
      *(_QWORD *)(a1 + 112));
    v22 = *(unsigned int **)(a1 + 48);
    v23 = &v22[4 * *(unsigned int *)(a1 + 56)];
    while ( v23 != v22 )
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *v22;
      v22 += 4;
      sub_B99FD0(v21, v25, v24);
    }
    return sub_9472D0(a1, v21, a2, a3, v36);
  }
  if ( *(_BYTE *)(a8 + 140) == 12 )
  {
    do
      v13 = *(_QWORD *)(v13 + 160);
    while ( *(_BYTE *)(v13 + 140) == 12 );
  }
  v28 = v34;
  BYTE1(v28) = v15;
  return sub_92CB70(a1, a2, a5, *(_QWORD *)(v13 + 128), v11, v28, v36 | a7);
}
