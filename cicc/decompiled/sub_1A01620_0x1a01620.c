// Function: sub_1A01620
// Address: 0x1a01620
//
void __fastcall sub_1A01620(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v6; // al
  __int64 v7; // rdi
  __int64 v8; // rax
  _QWORD *v9; // rax
  _BYTE *v10; // r13
  _BYTE *v11; // r14
  unsigned __int8 v12; // al
  __int64 v13; // rsi
  bool v14; // zf
  unsigned __int8 v15; // al
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v25; // [rsp+8h] [rbp-28h]

  *(_DWORD *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  v6 = *(_BYTE *)(a2 + 16);
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 32) = 0;
  if ( v6 > 0x17u && (unsigned __int8)(v6 - 50) <= 1u )
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v9 = *(_QWORD **)(a2 - 8);
    else
      v9 = (_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v10 = (_BYTE *)*v9;
    v11 = (_BYTE *)v9[3];
    v12 = *(_BYTE *)(*v9 + 16LL);
    v13 = (__int64)(v10 + 24);
    if ( v12 != 13 )
    {
      if ( *(_BYTE *)(*(_QWORD *)v10 + 8LL) == 16 && v12 <= 0x10u )
      {
        v18 = sub_15A1020(v10, v13, *(_QWORD *)v10, a4);
        if ( v18 )
        {
          if ( *(_BYTE *)(v18 + 16) == 13 )
          {
            v19 = v10;
            v10 = v11;
            v11 = v19;
          }
        }
      }
      v15 = v11[16];
      if ( v15 == 13 )
      {
        v17 = *(_DWORD *)(a1 + 24);
        v13 = (__int64)(v11 + 24);
      }
      else
      {
        if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) != 16 )
          goto LABEL_3;
        if ( v15 > 0x10u )
          goto LABEL_3;
        v16 = sub_15A1020(v11, v13, *(_QWORD *)v11, a4);
        if ( !v16 || *(_BYTE *)(v16 + 16) != 13 )
          goto LABEL_3;
        v13 = v16 + 24;
        v17 = *(_DWORD *)(a1 + 24);
      }
      v11 = v10;
      if ( v17 > 0x40 )
        goto LABEL_14;
    }
    if ( *(_DWORD *)(v13 + 8) <= 0x40u )
    {
      v20 = *(_QWORD *)v13;
      *(_QWORD *)(a1 + 16) = *(_QWORD *)v13;
      v21 = *(unsigned int *)(v13 + 8);
      *(_DWORD *)(a1 + 24) = v21;
      v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v21;
      if ( (unsigned int)v21 > 0x40 )
      {
        v23 = (unsigned int)((unsigned __int64)(v21 + 63) >> 6) - 1;
        *(_QWORD *)(v20 + 8 * v23) &= v22;
      }
      else
      {
        *(_QWORD *)(a1 + 16) = v22 & v20;
      }
    }
    else
    {
LABEL_14:
      sub_16A51C0(a1 + 16, v13);
    }
    v14 = *(_BYTE *)(a2 + 16) == 51;
    *(_QWORD *)(a1 + 8) = v11;
    *(_BYTE *)(a1 + 36) = v14;
    return;
  }
LABEL_3:
  *(_QWORD *)(a1 + 8) = a2;
  v25 = sub_16431D0(*(_QWORD *)a2);
  if ( v25 <= 0x40 )
    v24 = 0;
  else
    sub_16A4EF0((__int64)&v24, 0, 0);
  if ( *(_DWORD *)(a1 + 24) > 0x40u )
  {
    v7 = *(_QWORD *)(a1 + 16);
    if ( v7 )
      j_j___libc_free_0_0(v7);
  }
  v8 = v24;
  *(_BYTE *)(a1 + 36) = 1;
  *(_QWORD *)(a1 + 16) = v8;
  *(_DWORD *)(a1 + 24) = v25;
}
