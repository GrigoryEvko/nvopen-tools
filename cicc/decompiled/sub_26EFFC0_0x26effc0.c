// Function: sub_26EFFC0
// Address: 0x26effc0
//
__int64 __fastcall sub_26EFFC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // rax
  _QWORD *v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // r14
  bool v11; // zf
  __int64 *v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _BYTE *v16; // rax
  unsigned __int64 i; // rdi
  _BYTE *v18; // rdi
  __int64 v20; // [rsp+18h] [rbp-78h] BYREF
  _BYTE *v21; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v22; // [rsp+28h] [rbp-68h]
  _BYTE *v23; // [rsp+30h] [rbp-60h]
  _QWORD v24[2]; // [rsp+40h] [rbp-50h] BYREF
  void (__fastcall *v25)(_QWORD *, _QWORD *, __int64); // [rsp+50h] [rbp-40h]

  v3 = 69;
  v5 = sub_B6AC80(a3, 69);
  v21 = 0;
  v22 = 0;
  v23 = 0;
  if ( !v5 )
    goto LABEL_23;
  v6 = (_QWORD *)v5;
LABEL_4:
  while ( 2 )
  {
    v7 = v6[2];
    if ( v7 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v7 + 24);
        v9 = *(_QWORD *)(v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF));
        v10 = *(_QWORD *)(v8 + 32 * (1LL - (*(_DWORD *)(v8 + 4) & 0x7FFFFFF)));
        sub_B43D60((_QWORD *)v8);
        if ( *(_QWORD *)(v9 + 16) )
          goto LABEL_3;
        if ( *(_BYTE *)v9 > 0x15u )
          break;
        v20 = v9;
        v3 = (__int64)v22;
        if ( v22 == v23 )
        {
          sub_91DE00((__int64)&v21, v22, &v20);
          goto LABEL_3;
        }
        if ( v22 )
        {
          *(_QWORD *)v22 = v9;
          v3 = (__int64)v22;
        }
        v3 += 8;
        v11 = *(_QWORD *)(v10 + 16) == 0;
        v22 = (_BYTE *)v3;
        if ( !v11 )
          goto LABEL_4;
LABEL_11:
        if ( *(_BYTE *)v10 > 0x15u )
          goto LABEL_4;
        v24[0] = v10;
        v3 = (__int64)v22;
        if ( v22 == v23 )
        {
          sub_91DE00((__int64)&v21, v22, v24);
          goto LABEL_4;
        }
        if ( v22 )
        {
          *(_QWORD *)v22 = v10;
          v3 = (__int64)v22;
        }
        v7 = v6[2];
        v3 += 8;
        v22 = (_BYTE *)v3;
        if ( !v7 )
          goto LABEL_16;
      }
      v3 = 0;
      v20 = 0;
      v25 = 0;
      sub_F5CAB0((char *)v9, 0, 0, (__int64)v24);
      if ( v25 )
      {
        v3 = (__int64)v24;
        v25(v24, v24, 3);
      }
LABEL_3:
      if ( *(_QWORD *)(v10 + 16) )
        continue;
      goto LABEL_11;
    }
    break;
  }
LABEL_16:
  sub_B2E860(v6);
  v16 = v22;
  for ( i = (unsigned __int64)v21; v16 != v21; i = (unsigned __int64)v21 )
  {
    v18 = (_BYTE *)*((_QWORD *)v16 - 1);
    v16 -= 8;
    v22 = v16;
    if ( *v18 == 3 )
    {
      v12 = (__int64 *)((v18[32] & 0xFu) - 7);
      if ( (unsigned int)v12 > 1 )
        continue;
    }
    sub_26EFC50((__int64)v18, v3, v12, v13, v14, v15);
    v16 = v22;
  }
  if ( i )
    j_j___libc_free_0(i);
LABEL_23:
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_QWORD *)(a1 + 32) = &unk_4F82408;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  return a1;
}
