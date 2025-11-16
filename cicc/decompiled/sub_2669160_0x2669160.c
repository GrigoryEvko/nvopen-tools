// Function: sub_2669160
// Address: 0x2669160
//
void __fastcall sub_2669160(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // dl
  char v6; // cl
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned __int8 *v11; // rbx
  _QWORD *v12; // rdi
  __int64 v13; // r9
  unsigned __int8 *v14; // r14
  __int16 v15; // ax
  unsigned __int8 v16; // al
  __int16 v17; // dx
  unsigned __int8 v18; // dl
  unsigned __int8 v19; // al
  __int64 i; // rbx
  _BYTE *v21; // rdi
  unsigned __int64 v22; // rax
  __int16 v23; // dx
  unsigned __int8 v24; // [rsp+Fh] [rbp-61h] BYREF
  _BYTE v25[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  v5 = *(_BYTE *)(a3 + 32);
  v6 = v5 & 0xF;
  if ( ((((v5 & 0xF) + 9) & 0xFu) <= 1 || ((v6 + 15) & 0xFu) <= 2) && !*(_QWORD *)(a3 + 16) && !(_BYTE)qword_4FF4348 )
  {
LABEL_25:
    sub_B2E860((_QWORD *)a3);
    return;
  }
  if ( (_BYTE)qword_4FF4268 && v5 >> 6 == 2 )
  {
    v10 = *(_QWORD *)(a3 + 8);
    v11 = v25;
    v12 = *(_QWORD **)(a3 + 24);
    v13 = *(_QWORD *)(a3 + 40);
    v26 = 257;
    v14 = (unsigned __int8 *)sub_B30500(v12, *(_DWORD *)(v10 + 8) >> 8, v6, (__int64)v25, a2, v13);
    v15 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = (*(_WORD *)(a3 + 34) >> 1) & 0x3F;
      if ( !v17 )
      {
        v25[0] = 0;
        v11 = &v24;
        v24 = v16;
LABEL_17:
        sub_B2F770(a2, *v11);
        goto LABEL_18;
      }
      v18 = v17 - 1;
      v25[0] = v18;
    }
    else
    {
      v23 = (*(_WORD *)(a3 + 34) >> 1) & 0x3F;
      if ( !v23 )
      {
        sub_B2F740(a2, 0);
LABEL_18:
        sub_BD6B90(v14, (unsigned __int8 *)a3);
        v19 = *(_BYTE *)(a3 + 32) & 0x30 | v14[32] & 0xCF;
        v14[32] = v19;
        if ( (v19 & 0xFu) - 7 <= 1 || (v19 & 0x30) != 0 && (v19 & 0xF) != 9 )
          v14[33] |= 0x40u;
        v14[32] = v14[32] & 0x3F | 0x80;
        for ( i = *(_QWORD *)(a3 + 16); i; i = *(_QWORD *)(i + 8) )
        {
          v21 = *(_BYTE **)(i + 24);
          if ( *v21 > 0x1Cu )
          {
            v22 = sub_B43CB0((__int64)v21);
            sub_26673B0(a1, v22);
          }
        }
        sub_BD84D0(a3, (__int64)v14);
        goto LABEL_25;
      }
      v18 = v23 - 1;
      v16 = 0;
      v25[0] = v18;
    }
    v24 = v16;
    if ( v16 >= v18 )
      v11 = &v24;
    goto LABEL_17;
  }
  if ( !(*(_DWORD *)(*(_QWORD *)(a2 + 24) + 8LL) >> 8) )
  {
    v7 = *(_QWORD *)(a2 + 80);
    if ( a2 + 72 == v7 )
      goto LABEL_11;
    v8 = *(_QWORD *)(a2 + 80);
    v9 = 0;
    do
    {
      v8 = *(_QWORD *)(v8 + 8);
      ++v9;
    }
    while ( a2 + 72 != v8 );
    if ( v9 != 1 )
      goto LABEL_11;
    if ( v7 )
      v7 -= 24;
    if ( sub_AA6A60(v7) > 1 )
LABEL_11:
      sub_26677D0(a1, a2, a3);
  }
}
