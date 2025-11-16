// Function: sub_2A391C0
// Address: 0x2a391c0
//
void __fastcall sub_2A391C0(__int64 **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r13
  int v5; // eax
  __int64 v6; // rax
  __int64 v7; // r14
  char v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned int v13; // eax
  _BYTE *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-98h]
  __int64 v17; // [rsp+10h] [rbp-90h]
  char v18; // [rsp+1Fh] [rbp-81h]
  char v19; // [rsp+2Fh] [rbp-71h] BYREF
  _BYTE *v20; // [rsp+30h] [rbp-70h]
  __int64 v21; // [rsp+38h] [rbp-68h]
  __int64 v22; // [rsp+40h] [rbp-60h]
  _DWORD v23[22]; // [rsp+48h] [rbp-58h] BYREF

  v2 = *(_QWORD *)(a2 - 32);
  v20 = v23;
  v21 = 0;
  v22 = 32;
  v19 = 0;
  if ( !v2 || *(_BYTE *)v2 || *(_QWORD *)(v2 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  switch ( *(_DWORD *)(v2 + 36) )
  {
    case 0xEE:
      qmemcpy(v23, "memcpy", 6);
      v21 = 6;
      v18 = 0;
      goto LABEL_6;
    case 0xEF:
      qmemcpy(v23, "memcpy", 6);
      v21 = 6;
      v18 = 1;
      goto LABEL_6;
    case 0xF0:
      qmemcpy(v23, "memcpy", 6);
      v21 = 6;
      v19 = 1;
      v18 = 0;
      goto LABEL_6;
    case 0xF1:
      qmemcpy(v23, "memmove", 7);
      v21 = 7;
      v18 = 0;
      goto LABEL_6;
    case 0xF2:
      qmemcpy(v23, "memmove", 7);
      v21 = 7;
      v18 = 1;
      goto LABEL_6;
    case 0xF3:
      qmemcpy(v23, "memset", 6);
      v21 = 6;
      v18 = 0;
      goto LABEL_6;
    case 0xF4:
      qmemcpy(v23, "memset", 6);
      v21 = 6;
      v18 = 1;
LABEL_6:
      v16 = ((__int64 (__fastcall *)(__int64 **, __int64))(*a1)[3])(a1, 2);
      v4 = v3;
      v17 = (__int64)a1[2];
      v5 = ((__int64 (__fastcall *)(__int64 **))(*a1)[4])(a1);
      if ( v5 == 14 )
      {
        v6 = sub_22077B0(0x1B0u);
        v7 = v6;
        if ( v6 )
          sub_B176B0(v6, v17, v16, v4, a2);
      }
      else
      {
        if ( v5 != 15 )
          BUG();
        v15 = sub_22077B0(0x1B0u);
        v7 = v15;
        if ( v15 )
          sub_B178C0(v15, v17, v16, v4, a2);
      }
      v8 = 0;
      sub_2A39010((__int64)a1, v20, v21, 1, v7);
      sub_2A38760((__int64)a1, *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), v7);
      v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
      v10 = *(_QWORD *)(a2 + 32 * (3 - v9));
      if ( *(_BYTE *)v10 == 17 && !v18 )
      {
        if ( *(_DWORD *)(v10 + 32) <= 0x40u )
          v11 = *(_QWORD *)(v10 + 24);
        else
          v11 = **(_QWORD **)(v10 + 24);
        v8 = v11 != 0;
      }
      v12 = *(_QWORD *)(a2 - 32);
      if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != *(_QWORD *)(a2 + 80) )
        BUG();
      v13 = *(_DWORD *)(v12 + 36);
      if ( v13 > 0xF1 )
      {
        if ( v13 - 243 <= 1 )
          sub_2A38830((__int64)a1, *(unsigned __int8 **)(a2 - 32 * v9), 0, v7);
      }
      else if ( v13 > 0xED )
      {
        sub_2A38830((__int64)a1, *(unsigned __int8 **)(a2 + 32 * (1 - v9)), 1, v7);
        sub_2A38830((__int64)a1, *(unsigned __int8 **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)), 0, v7);
      }
      sub_2A381E0(&v19, v8, v18, v7);
      sub_1049740(a1[1], v7);
      if ( v7 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 16LL))(v7);
      v14 = v20;
      if ( v20 != (_BYTE *)v23 )
        goto LABEL_23;
      return;
    default:
      sub_2A37680(a1, a2);
      v14 = v20;
      if ( v20 == (_BYTE *)v23 )
        return;
LABEL_23:
      _libc_free((unsigned __int64)v14);
      return;
  }
}
