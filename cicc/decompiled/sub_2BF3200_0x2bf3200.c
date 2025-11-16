// Function: sub_2BF3200
// Address: 0x2bf3200
//
void __fastcall sub_2BF3200(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  unsigned __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __int64 *v12; // r9
  _QWORD *v13; // rax
  __int64 *v14; // r9
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // r15
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 *v23; // [rsp+8h] [rbp-68h]
  __int64 *v24; // [rsp+8h] [rbp-68h]
  __int64 v25[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 128);
  v5 = *(_QWORD *)(a2 + 904);
  v6 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v6 == v4 + 48 )
  {
    v8 = 0;
  }
  else
  {
    if ( !v6 )
      BUG();
    v7 = *(unsigned __int8 *)(v6 - 24);
    v8 = v6 - 24;
    if ( (unsigned int)(v7 - 30) >= 0xB )
      v8 = 0;
  }
  sub_D5F1F0(v5, v8);
  v9 = *(_QWORD *)(a1 + 128);
  v25[0] = a1;
  *(_QWORD *)(a2 + 104) = v9;
  *sub_2BF2B80(a2 + 120, v25) = v9;
  sub_2BF0980(a1, a2);
  if ( *(_DWORD *)(a1 + 88) == 1 && **(_QWORD **)(a1 + 80) )
  {
    v10 = *(_QWORD *)(a1 + 128);
    v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11 == v10 + 48 )
      goto LABEL_21;
    if ( !v11 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v11 - 24) - 30 > 0xA )
LABEL_21:
      BUG();
    if ( *(_BYTE *)(v11 - 24) == 36 )
    {
      v12 = *(__int64 **)(a2 + 904);
      v26 = 257;
      v23 = v12;
      v13 = sub_BD2C40(72, 1u);
      v14 = v23;
      v15 = (__int64)v13;
      if ( v13 )
      {
        sub_B4C8F0((__int64)v13, v10, 1u, 0, 0);
        v14 = v23;
      }
      v24 = v14;
      (*(void (__fastcall **)(__int64, __int64, __int64 *, __int64, __int64))(*(_QWORD *)v14[11] + 16LL))(
        v14[11],
        v15,
        v25,
        v14[7],
        v14[8]);
      v16 = *v24;
      v17 = *v24 + 16LL * *((unsigned int *)v24 + 2);
      if ( *v24 != v17 )
      {
        do
        {
          v18 = *(_QWORD *)(v16 + 8);
          v19 = *(_DWORD *)v16;
          v16 += 16;
          sub_B99FD0(v15, v19, v18);
        }
        while ( v17 != v16 );
      }
      v20 = v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v20 )
      {
        v21 = *(_QWORD *)(v20 + 8);
        **(_QWORD **)(v20 + 16) = v21;
        if ( v21 )
          *(_QWORD *)(v21 + 16) = *(_QWORD *)(v20 + 16);
      }
      *(_QWORD *)v20 = 0;
      v22 = (_QWORD *)sub_986580(*(_QWORD *)(a1 + 128));
      sub_B43D60(v22);
    }
  }
  sub_2BF2D10((__int64 *)a1, a2);
}
