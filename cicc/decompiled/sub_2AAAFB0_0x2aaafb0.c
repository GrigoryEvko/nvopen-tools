// Function: sub_2AAAFB0
// Address: 0x2aaafb0
//
__int64 __fastcall sub_2AAAFB0(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r13
  __int64 v4; // r15
  __int64 v5; // r8
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v16; // [rsp+0h] [rbp-B0h]
  __int64 v17; // [rsp+0h] [rbp-B0h]
  int v18; // [rsp+14h] [rbp-9Ch]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  __int64 v21; // [rsp+28h] [rbp-88h] BYREF
  _QWORD v22[3]; // [rsp+30h] [rbp-80h] BYREF
  char v23; // [rsp+48h] [rbp-68h] BYREF
  void *v24; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v1 = *(__int64 **)(a1 + 48);
  v2 = *v1;
  v3 = v1[1];
  v4 = v1[2];
  v25 = 257;
  v6 = sub_22077B0(0xC8u);
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 160);
    v22[2] = v4;
    v8 = v6 + 40;
    v22[1] = v3;
    v9 = v6 + 64;
    v10 = 0;
    v19 = v7;
    LODWORD(v7) = *(_DWORD *)(a1 + 152);
    v11 = (char *)v22;
    v20 = 0;
    v18 = v7;
    v22[0] = v2;
    *(_BYTE *)(v6 + 8) = 1;
    *(_QWORD *)v6 = &unk_4A231A8;
    v21 = 0;
    *(_QWORD *)(v6 + 48) = v6 + 64;
    *(_QWORD *)(v6 + 40) = &unk_4A23170;
    *(_QWORD *)(v6 + 56) = 0x200000000LL;
    *(_QWORD *)(v6 + 24) = 0;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 16) = 0;
    while ( 1 )
    {
      *(_QWORD *)(v9 + 8 * v10) = v2;
      ++*(_DWORD *)(v6 + 56);
      v12 = *(unsigned int *)(v2 + 24);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(v2 + 28) )
      {
        v16 = v8;
        sub_C8D5F0(v2 + 16, (const void *)(v2 + 32), v12 + 1, 8u, v5, v8);
        v12 = *(unsigned int *)(v2 + 24);
        v8 = v16;
      }
      v11 += 8;
      *(_QWORD *)(*(_QWORD *)(v2 + 16) + 8 * v12) = v8;
      ++*(_DWORD *)(v2 + 24);
      if ( v11 == &v23 )
        break;
      v10 = *(unsigned int *)(v6 + 56);
      v2 = *(_QWORD *)v11;
      if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 60) )
      {
        v17 = v8;
        sub_C8D5F0(v6 + 48, (const void *)(v6 + 64), v10 + 1, 8u, v5, v8);
        v10 = *(unsigned int *)(v6 + 56);
        v8 = v17;
      }
      v9 = *(_QWORD *)(v6 + 48);
    }
    *(_QWORD *)(v6 + 80) = 0;
    *(_QWORD *)(v6 + 40) = &unk_4A23AA8;
    v13 = v21;
    *(_QWORD *)v6 = &unk_4A23A70;
    *(_QWORD *)(v6 + 88) = v13;
    if ( v13 )
    {
      sub_2AAAFA0((__int64 *)(v6 + 88));
      if ( v21 )
        sub_B91220((__int64)&v21, v21);
    }
    sub_2BF0340(v6 + 96, 1, 0, v6);
    v14 = v20;
    *(_QWORD *)v6 = &unk_4A231C8;
    *(_QWORD *)(v6 + 40) = &unk_4A23200;
    *(_QWORD *)(v6 + 96) = &unk_4A23238;
    if ( v14 )
      sub_B91220((__int64)&v20, v14);
    *(_QWORD *)v6 = &unk_4A23718;
    *(_QWORD *)(v6 + 96) = &unk_4A23790;
    *(_QWORD *)(v6 + 40) = &unk_4A23758;
    *(_DWORD *)(v6 + 152) = v18;
    *(_QWORD *)(v6 + 160) = v19;
    sub_CA0F50((__int64 *)(v6 + 168), &v24);
  }
  return v6;
}
