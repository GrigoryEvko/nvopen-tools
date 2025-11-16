// Function: sub_2DCFB00
// Address: 0x2dcfb00
//
__int64 __fastcall sub_2DCFB00(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 **v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void **v10; // rax
  __int64 **v12; // rsi
  __int64 v13; // [rsp+0h] [rbp-70h] BYREF
  void **v14; // [rsp+8h] [rbp-68h]
  unsigned int v15; // [rsp+10h] [rbp-60h]
  unsigned int v16; // [rsp+14h] [rbp-5Ch]
  char v17; // [rsp+1Ch] [rbp-54h]
  _BYTE v18[16]; // [rsp+20h] [rbp-50h] BYREF
  _BYTE v19[8]; // [rsp+30h] [rbp-40h] BYREF
  unsigned __int64 v20; // [rsp+38h] [rbp-38h]
  int v21; // [rsp+44h] [rbp-2Ch]
  int v22; // [rsp+48h] [rbp-28h]
  char v23; // [rsp+4Ch] [rbp-24h]
  _BYTE v24[32]; // [rsp+50h] [rbp-20h] BYREF

  if ( (*(_BYTE *)(*a3 + 3LL) & 0x40) == 0 || !(unsigned __int8)sub_2DCCA90(a3, a2, (__int64)a3, a4, a5, a6) )
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0(&v13);
  if ( v21 != v22 )
    goto LABEL_4;
  if ( v17 )
  {
    v10 = v14;
    v12 = (__int64 **)&v14[v16];
    v7 = v16;
    v6 = (__int64 **)v14;
    if ( v14 == (void **)v12 )
      goto LABEL_21;
    while ( *v6 != &qword_4F82400 )
    {
      if ( v12 == ++v6 )
      {
LABEL_8:
        while ( *v10 != &unk_4F82408 )
        {
          if ( ++v10 == (void **)v6 )
            goto LABEL_21;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v13, (__int64)&qword_4F82400) )
  {
LABEL_4:
    if ( !v17 )
    {
LABEL_23:
      sub_C8CC70((__int64)&v13, (__int64)&unk_4F82408, (__int64)v6, v7, v8, v9);
      goto LABEL_9;
    }
    v10 = v14;
    v7 = v16;
    v6 = (__int64 **)&v14[v16];
    if ( v14 != (void **)v6 )
      goto LABEL_8;
LABEL_21:
    if ( v15 > (unsigned int)v7 )
    {
      v16 = v7 + 1;
      *v6 = (__int64 *)&unk_4F82408;
      ++v13;
      goto LABEL_9;
    }
    goto LABEL_23;
  }
LABEL_9:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v18, (__int64)&v13);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v24, (__int64)v19);
  if ( !v23 )
    _libc_free(v20);
  if ( !v17 )
  {
    _libc_free((unsigned __int64)v14);
    return a1;
  }
  return a1;
}
