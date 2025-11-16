// Function: sub_2F7E0A0
// Address: 0x2f7e0a0
//
__int64 __fastcall sub_2F7E0A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 *v9; // r14
  char v10; // r15
  char v11; // bl
  char v12; // al
  __int64 **v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  void **v17; // rax
  __int64 **v19; // rdi
  __int64 v20; // [rsp+0h] [rbp-90h] BYREF
  void **v21; // [rsp+8h] [rbp-88h]
  unsigned int v22; // [rsp+10h] [rbp-80h]
  unsigned int v23; // [rsp+14h] [rbp-7Ch]
  char v24; // [rsp+1Ch] [rbp-74h]
  _BYTE v25[16]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v26[8]; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v27; // [rsp+38h] [rbp-58h]
  int v28; // [rsp+44h] [rbp-4Ch]
  int v29; // [rsp+48h] [rbp-48h]
  char v30; // [rsp+4Ch] [rbp-44h]
  _BYTE v31[64]; // [rsp+50h] [rbp-40h] BYREF

  if ( !sub_B92180(*a3) )
    goto LABEL_18;
  v5 = sub_B92180(*a3);
  v6 = *(_BYTE *)(v5 - 16);
  v7 = (v6 & 2) != 0 ? *(_QWORD *)(v5 - 32) : v5 - 16 - 8LL * ((v6 >> 2) & 0xF);
  if ( !*(_DWORD *)(*(_QWORD *)(v7 + 40) + 32LL) )
    goto LABEL_18;
  v8 = a3[41];
  v9 = a3 + 40;
  v10 = 0;
  if ( (__int64 *)v8 == v9 )
    goto LABEL_18;
  do
  {
    v11 = sub_2F7D880(v8);
    v12 = sub_2F7D060(v8);
    v8 = *(_QWORD *)(v8 + 8);
    v10 |= v12 | v11;
  }
  while ( v9 != (__int64 *)v8 );
  if ( !v10 )
  {
LABEL_18:
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
  sub_2EAFFB0((__int64)&v20);
  if ( v28 == v29 )
  {
    if ( v24 )
    {
      v17 = v21;
      v19 = (__int64 **)&v21[v23];
      v14 = v23;
      v13 = (__int64 **)v21;
      if ( v21 != (void **)v19 )
      {
        while ( *v13 != &qword_4F82400 )
        {
          if ( v19 == ++v13 )
          {
LABEL_13:
            while ( *v17 != &unk_4F82408 )
            {
              if ( ++v17 == (void **)v13 )
                goto LABEL_21;
            }
            goto LABEL_14;
          }
        }
        goto LABEL_14;
      }
      goto LABEL_21;
    }
    if ( sub_C8CA60((__int64)&v20, (__int64)&qword_4F82400) )
      goto LABEL_14;
  }
  if ( !v24 )
  {
LABEL_23:
    sub_C8CC70((__int64)&v20, (__int64)&unk_4F82408, (__int64)v13, v14, v15, v16);
    goto LABEL_14;
  }
  v17 = v21;
  v14 = v23;
  v13 = (__int64 **)&v21[v23];
  if ( v21 != (void **)v13 )
    goto LABEL_13;
LABEL_21:
  if ( (unsigned int)v14 >= v22 )
    goto LABEL_23;
  v23 = v14 + 1;
  *v13 = (__int64 *)&unk_4F82408;
  ++v20;
LABEL_14:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v25, (__int64)&v20);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v31, (__int64)v26);
  if ( !v30 )
    _libc_free(v27);
  if ( !v24 )
    _libc_free((unsigned __int64)v21);
  return a1;
}
