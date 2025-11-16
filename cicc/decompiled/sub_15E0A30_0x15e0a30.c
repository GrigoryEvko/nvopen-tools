// Function: sub_15E0A30
// Address: 0x15e0a30
//
void __fastcall sub_15E0A30(__int64 a1, __int64 a2)
{
  __int64 v3; // r8
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  size_t v7; // rdx
  const void *v8; // r9
  _BYTE *v9; // rdi
  const void *v10; // [rsp+0h] [rbp-100h]
  size_t n; // [rsp+18h] [rbp-E8h]
  int na; // [rsp+18h] [rbp-E8h]
  _QWORD v13[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v14; // [rsp+30h] [rbp-D0h]
  _BYTE *v15; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+48h] [rbp-B8h]
  _BYTE v17[176]; // [rsp+50h] [rbp-B0h] BYREF

  if ( (*(_BYTE *)(a1 + 18) & 1) == 0 )
  {
    sub_15E09A0(a1);
    *(_WORD *)(a1 + 18) |= 1u;
  }
  if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
  {
    *(_QWORD *)(a1 + 88) = *(_QWORD *)(a2 + 88);
    *(_QWORD *)(a2 + 88) = 0;
    v3 = *(_QWORD *)(a1 + 88);
    v4 = v3 + 40LL * *(_QWORD *)(a1 + 96);
    v5 = v3;
    if ( v3 == v4 )
    {
LABEL_15:
      *(_WORD *)(a1 + 18) &= ~1u;
      *(_WORD *)(a2 + 18) |= 1u;
      return;
    }
    while ( 1 )
    {
      v15 = v17;
      v16 = 0x8000000000LL;
      if ( (*(_BYTE *)(v5 + 23) & 0x20) != 0 )
      {
        v6 = sub_1649960(v5);
        LODWORD(v16) = 0;
        v8 = (const void *)v6;
        if ( v7 > HIDWORD(v16) )
        {
          v10 = (const void *)v6;
          n = v7;
          sub_16CD150(&v15, v17, v7, 1);
          v7 = n;
          v8 = v10;
          v9 = &v15[(unsigned int)v16];
LABEL_18:
          na = v7;
          memcpy(v9, v8, v7);
          LODWORD(v16) = v16 + na;
          if ( (_DWORD)v16 )
          {
            v14 = 257;
            sub_164B780(v5, v13);
          }
          goto LABEL_6;
        }
        if ( v7 )
        {
          v9 = v15;
          goto LABEL_18;
        }
      }
LABEL_6:
      sub_15E02C0(v5, a1);
      if ( (_DWORD)v16 )
      {
        v14 = 262;
        v13[0] = &v15;
        sub_164B780(v5, v13);
      }
      if ( v15 != v17 )
        _libc_free((unsigned __int64)v15);
      v5 += 40;
      if ( v4 == v5 )
        goto LABEL_15;
    }
  }
}
