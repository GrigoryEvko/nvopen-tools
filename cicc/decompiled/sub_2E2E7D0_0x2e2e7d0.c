// Function: sub_2E2E7D0
// Address: 0x2e2e7d0
//
__int64 __fastcall sub_2E2E7D0(__int64 a1, __int64 a2, __int64 a3)
{
  char v3; // bl
  __int64 **v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  void **v9; // rax
  __int64 **v10; // rsi
  unsigned int *v11; // [rsp+0h] [rbp-C0h] BYREF
  unsigned __int64 v12; // [rsp+8h] [rbp-B8h]
  unsigned int v13; // [rsp+10h] [rbp-B0h] BYREF
  unsigned int v14; // [rsp+14h] [rbp-ACh]
  char v15; // [rsp+1Ch] [rbp-A4h]
  _BYTE v16[16]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE v17[8]; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v18; // [rsp+38h] [rbp-88h]
  int v19; // [rsp+44h] [rbp-7Ch]
  int v20; // [rsp+48h] [rbp-78h]
  char v21; // [rsp+4Ch] [rbp-74h]
  _BYTE v22[112]; // [rsp+50h] [rbp-70h] BYREF

  v11 = &v13;
  v12 = 0x1000000000LL;
  v3 = sub_2E2E630((__int64)&v11, a3);
  if ( v11 != &v13 )
    _libc_free((unsigned __int64)v11);
  if ( !v3 )
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
  sub_2EAFFB0(&v11);
  if ( v19 == v20 )
  {
    if ( v15 )
    {
      v9 = (void **)v12;
      v10 = (__int64 **)(v12 + 8LL * v14);
      v6 = v14;
      v5 = (__int64 **)v12;
      if ( (__int64 **)v12 != v10 )
      {
        while ( *v5 != &qword_4F82400 )
        {
          if ( v10 == ++v5 )
          {
LABEL_11:
            while ( *v9 != &unk_4F82408 )
            {
              if ( v5 == (__int64 **)++v9 )
                goto LABEL_16;
            }
            goto LABEL_12;
          }
        }
        goto LABEL_12;
      }
      goto LABEL_16;
    }
    if ( sub_C8CA60((__int64)&v11, (__int64)&qword_4F82400) )
      goto LABEL_12;
  }
  if ( !v15 )
  {
LABEL_18:
    sub_C8CC70((__int64)&v11, (__int64)&unk_4F82408, (__int64)v5, v6, v7, v8);
    goto LABEL_12;
  }
  v9 = (void **)v12;
  v6 = v14;
  v5 = (__int64 **)(v12 + 8LL * v14);
  if ( (__int64 **)v12 != v5 )
    goto LABEL_11;
LABEL_16:
  if ( (unsigned int)v6 >= v13 )
    goto LABEL_18;
  v14 = v6 + 1;
  *v5 = (__int64 *)&unk_4F82408;
  v11 = (unsigned int *)((char *)v11 + 1);
LABEL_12:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v16, (__int64)&v11);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v22, (__int64)v17);
  if ( !v21 )
    _libc_free(v18);
  if ( !v15 )
    _libc_free(v12);
  return a1;
}
