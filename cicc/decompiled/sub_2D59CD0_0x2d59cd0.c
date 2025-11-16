// Function: sub_2D59CD0
// Address: 0x2d59cd0
//
void __fastcall sub_2D59CD0(unsigned __int64 a1)
{
  unsigned __int64 v2; // r14
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // r15
  __int64 v10; // rax
  _QWORD v11[2]; // [rsp+8h] [rbp-88h] BYREF
  __int64 v12; // [rsp+18h] [rbp-78h]
  __int64 v13; // [rsp+20h] [rbp-70h]
  void *v14; // [rsp+30h] [rbp-60h]
  _QWORD v15[2]; // [rsp+38h] [rbp-58h] BYREF
  __int64 v16; // [rsp+48h] [rbp-48h]
  __int64 v17; // [rsp+50h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 176) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 184), 24LL * *(unsigned int *)(a1 + 192), 8);
  if ( (*(_BYTE *)(a1 + 96) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 104), 16LL * *(unsigned int *)(a1 + 112), 8);
  v2 = *(_QWORD *)(a1 + 80);
  if ( v2 )
  {
    v3 = *(_QWORD *)(v2 + 40);
    v4 = *(_QWORD *)(v2 + 32);
    if ( v3 != v4 )
    {
      do
      {
        v5 = *(unsigned int *)(v4 + 24);
        v6 = *(_QWORD *)(v4 + 8);
        v4 += 32LL;
        sub_C7D6A0(v6, 16 * v5, 8);
      }
      while ( v3 != v4 );
      v4 = *(_QWORD *)(v2 + 32);
    }
    if ( v4 )
      j_j___libc_free_0(v4);
    sub_C7D6A0(*(_QWORD *)(v2 + 8), 16LL * *(unsigned int *)(v2 + 24), 8);
    j_j___libc_free_0(v2);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), 24LL * *(unsigned int *)(a1 + 56), 8);
  v7 = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD **)(a1 + 8);
    v11[0] = 2;
    v11[1] = 0;
    v12 = -4096;
    v9 = &v8[5 * v7];
    v13 = 0;
    v15[0] = 2;
    v15[1] = 0;
    v16 = -8192;
    v14 = &unk_49DE380;
    v17 = 0;
    do
    {
      v10 = v8[3];
      *v8 = &unk_49DB368;
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0(v8 + 1);
      v8 += 5;
    }
    while ( v9 != v8 );
    v14 = &unk_49DB368;
    if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
      sub_BD60C0(v15);
    if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
      sub_BD60C0(v11);
    v7 = *(unsigned int *)(a1 + 24);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 40 * v7, 8);
  j_j___libc_free_0(a1);
}
