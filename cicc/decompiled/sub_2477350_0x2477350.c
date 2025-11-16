// Function: sub_2477350
// Address: 0x2477350
//
void __fastcall sub_2477350(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  __int64 v4; // r13
  unsigned __int8 *v5; // r14
  __int64 v6; // rsi
  __int64 v7; // r10
  __int64 v8; // rax
  unsigned __int8 v9; // dl
  __int64 v10; // rax
  __int64 v11; // r10
  _BYTE *v12; // r14
  __int64 i; // rdi
  unsigned __int64 v14; // rax
  _BYTE *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v20; // [rsp+8h] [rbp-108h]
  __int64 v21; // [rsp+8h] [rbp-108h]
  __int64 v22; // [rsp+8h] [rbp-108h]
  __int64 v23; // [rsp+8h] [rbp-108h]
  __int64 v24; // [rsp+8h] [rbp-108h]
  __int64 *v25; // [rsp+18h] [rbp-F8h]
  _BYTE v26[32]; // [rsp+20h] [rbp-F0h] BYREF
  __int16 v27; // [rsp+40h] [rbp-D0h]
  unsigned int *v28[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v29; // [rsp+60h] [rbp-B0h] BYREF
  void *v30; // [rsp+D0h] [rbp-40h]

  sub_23D0AB0((__int64)v28, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v3 = *(__int64 **)(a2 - 8);
    v25 = &v3[4 * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)];
  }
  else
  {
    v25 = (__int64 *)a2;
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  }
  v4 = 0;
  if ( v3 != v25 )
  {
    while ( 1 )
    {
      v5 = (unsigned __int8 *)*v3;
      v6 = *v3;
      v7 = sub_246F3F0(a1, *v3);
      v8 = *(_QWORD *)(a1 + 8);
      if ( !*(_DWORD *)(v8 + 4) )
        goto LABEL_5;
      if ( *(_BYTE *)(a1 + 633) && (v9 = *v5, *v5 > 0x15u) && v9 != 25 )
      {
        if ( v9 > 0x1Cu && (v5[7] & 0x20) != 0 && (v23 = v7, v16 = sub_B91C10((__int64)v5, 31), v7 = v23, v16) )
        {
          v17 = sub_AD6530(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 88LL), 31);
          v11 = v23;
          v12 = (_BYTE *)v17;
        }
        else
        {
          v24 = v7;
          v18 = sub_246EC10(a1 + 384, (__int64)v5);
          v11 = v24;
          v12 = (_BYTE *)*v18;
        }
      }
      else
      {
        v20 = v7;
        v10 = sub_AD6530(*(_QWORD *)(v8 + 88), v6);
        v11 = v20;
        v12 = (_BYTE *)v10;
      }
      if ( !*(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
        goto LABEL_5;
      if ( !v4 )
        break;
      if ( *v12 > 0x15u )
      {
LABEL_16:
        v27 = 257;
        for ( i = *(_QWORD *)(v11 + 8); *(_BYTE *)(i + 8) != 12; v11 = v14 )
        {
          v14 = sub_24650D0(a1, v11, (__int64)v28);
          i = *(_QWORD *)(v14 + 8);
        }
        if ( *(_DWORD *)(i + 8) >> 8 != 1 )
        {
          v22 = v11;
          v15 = (_BYTE *)sub_AD64C0(i, 0, 0);
          v11 = sub_92B530(v28, 0x21u, v22, v15, (__int64)v26);
        }
        v27 = 257;
        v3 += 4;
        v4 = sub_B36550(v28, v11, (__int64)v12, v4, (__int64)v26, 0);
        if ( v25 == v3 )
          goto LABEL_21;
      }
      else
      {
        v21 = v11;
        if ( !sub_AC30F0((__int64)v12) )
        {
          v11 = v21;
          goto LABEL_16;
        }
LABEL_5:
        v3 += 4;
        if ( v25 == v3 )
          goto LABEL_21;
      }
    }
    v4 = (__int64)v12;
    goto LABEL_5;
  }
LABEL_21:
  if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + 4LL) )
    sub_246F1C0(a1, a2, v4);
  nullsub_61();
  v30 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v28[0] != &v29 )
    _libc_free((unsigned __int64)v28[0]);
}
