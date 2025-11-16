// Function: sub_2DCA220
// Address: 0x2dca220
//
__int64 __fastcall sub_2DCA220(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int16 v3; // ax
  unsigned __int16 v5; // bx
  __int64 **v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  void **v10; // rax
  __int64 **v11; // rsi
  __int64 v12; // [rsp+0h] [rbp-90h] BYREF
  void **v13; // [rsp+8h] [rbp-88h]
  unsigned int v14; // [rsp+10h] [rbp-80h]
  unsigned int v15; // [rsp+14h] [rbp-7Ch]
  char v16; // [rsp+1Ch] [rbp-74h]
  _BYTE v17[16]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v18[8]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v19; // [rsp+38h] [rbp-58h]
  int v20; // [rsp+44h] [rbp-4Ch]
  int v21; // [rsp+48h] [rbp-48h]
  char v22; // [rsp+4Ch] [rbp-44h]
  _BYTE v23[64]; // [rsp+50h] [rbp-40h] BYREF

  v3 = sub_2DC9FD0(a3);
  if ( !(_BYTE)v3 )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
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
  v5 = v3;
  sub_2EAFFB0(&v12);
  if ( HIBYTE(v5) )
  {
    if ( v20 == v21 )
    {
      if ( v16 )
      {
        v10 = v13;
        v11 = (__int64 **)&v13[v15];
        v7 = v15;
        v6 = (__int64 **)v13;
        if ( v13 != (void **)v11 )
        {
          while ( *v6 != &qword_4F82400 )
          {
            if ( v11 == ++v6 )
            {
LABEL_12:
              while ( *v10 != &unk_4F82408 )
              {
                if ( ++v10 == (void **)v6 )
                  goto LABEL_14;
              }
              goto LABEL_5;
            }
          }
          goto LABEL_5;
        }
        goto LABEL_14;
      }
      if ( sub_C8CA60((__int64)&v12, (__int64)&qword_4F82400) )
        goto LABEL_5;
    }
    if ( !v16 )
      goto LABEL_16;
    v10 = v13;
    v7 = v15;
    v6 = (__int64 **)&v13[v15];
    if ( v13 != (void **)v6 )
      goto LABEL_12;
LABEL_14:
    if ( v14 > (unsigned int)v7 )
    {
      v15 = v7 + 1;
      *v6 = (__int64 *)&unk_4F82408;
      ++v12;
      goto LABEL_5;
    }
LABEL_16:
    sub_C8CC70((__int64)&v12, (__int64)&unk_4F82408, (__int64)v6, v7, v8, v9);
  }
LABEL_5:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v17, (__int64)&v12);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v23, (__int64)v18);
  if ( !v22 )
    _libc_free(v19);
  if ( !v16 )
    _libc_free((unsigned __int64)v13);
  return a1;
}
