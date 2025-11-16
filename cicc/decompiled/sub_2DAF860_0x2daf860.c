// Function: sub_2DAF860
// Address: 0x2daf860
//
__int64 __fastcall sub_2DAF860(__int64 a1, __int64 a2, __int64 a3)
{
  void *v3; // r14
  __int64 **v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  void **v9; // rax
  __int64 **v10; // rsi
  __int64 v11; // [rsp+0h] [rbp-80h] BYREF
  void **v12; // [rsp+8h] [rbp-78h]
  unsigned int v13; // [rsp+10h] [rbp-70h]
  unsigned int v14; // [rsp+14h] [rbp-6Ch]
  char v15; // [rsp+1Ch] [rbp-64h]
  _BYTE v16[16]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v17[8]; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v18; // [rsp+38h] [rbp-48h]
  int v19; // [rsp+44h] [rbp-3Ch]
  int v20; // [rsp+48h] [rbp-38h]
  char v21; // [rsp+4Ch] [rbp-34h]
  _BYTE v22[48]; // [rsp+50h] [rbp-30h] BYREF

  v3 = (void *)(a1 + 32);
  v11 = 0;
  v12 = 0;
  if ( !(unsigned __int8)sub_2DAF430(&v11, a3) )
  {
    *(_QWORD *)(a1 + 8) = v3;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  sub_2EAFFB0(&v11);
  if ( v19 != v20 )
    goto LABEL_5;
  if ( v15 )
  {
    v9 = v12;
    v10 = (__int64 **)&v12[v14];
    v6 = v14;
    v5 = (__int64 **)v12;
    if ( v12 == (void **)v10 )
      goto LABEL_14;
    while ( *v5 != &qword_4F82400 )
    {
      if ( v10 == ++v5 )
      {
LABEL_9:
        while ( *v9 != &unk_4F82408 )
        {
          if ( ++v9 == (void **)v5 )
            goto LABEL_14;
        }
        break;
      }
    }
  }
  else if ( !sub_C8CA60((__int64)&v11, (__int64)&qword_4F82400) )
  {
LABEL_5:
    if ( !v15 )
    {
LABEL_16:
      sub_C8CC70((__int64)&v11, (__int64)&unk_4F82408, (__int64)v5, v6, v7, v8);
      goto LABEL_10;
    }
    v9 = v12;
    v6 = v14;
    v5 = (__int64 **)&v12[v14];
    if ( v12 != (void **)v5 )
      goto LABEL_9;
LABEL_14:
    if ( (unsigned int)v6 < v13 )
    {
      v14 = v6 + 1;
      *v5 = (__int64 *)&unk_4F82408;
      ++v11;
      goto LABEL_10;
    }
    goto LABEL_16;
  }
LABEL_10:
  sub_C8CF70(a1, v3, 2, (__int64)v16, (__int64)&v11);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v22, (__int64)v17);
  if ( !v21 )
    _libc_free(v18);
  if ( v15 )
    return a1;
  _libc_free((unsigned __int64)v12);
  return a1;
}
