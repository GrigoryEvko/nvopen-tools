// Function: sub_287D200
// Address: 0x287d200
//
__int64 __fastcall sub_287D200(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5)
{
  __int64 *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 **v14; // rdx
  __int64 v15; // rcx
  void **v16; // rax
  int v17; // eax
  __int64 *v18; // rax
  void **v19; // rax
  __int64 **v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-90h] BYREF
  void **v22; // [rsp+8h] [rbp-88h]
  unsigned int v23; // [rsp+10h] [rbp-80h]
  unsigned int v24; // [rsp+14h] [rbp-7Ch]
  char v25; // [rsp+1Ch] [rbp-74h]
  _BYTE v26[16]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v27; // [rsp+30h] [rbp-60h] BYREF
  void **v28; // [rsp+38h] [rbp-58h]
  unsigned int v29; // [rsp+44h] [rbp-4Ch]
  int v30; // [rsp+48h] [rbp-48h]
  char v31; // [rsp+4Ch] [rbp-44h]
  _BYTE v32[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = (__int64 *)a5[4];
  if ( !(unsigned __int8)sub_287C150(a3, v6, a5[2], a5[6], a5[5], a5[9]) )
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
  sub_22D0390((__int64)&v21, (__int64)v6, v7, v8, v9, v10);
  if ( a5[9] )
  {
    if ( v31 )
    {
      v14 = (__int64 **)&v28[v29];
      v15 = v29;
      if ( v28 == (void **)v14 )
      {
LABEL_25:
        v17 = v30;
      }
      else
      {
        v16 = v28;
        while ( *v16 != &unk_4F8F810 )
        {
          if ( v14 == (__int64 **)++v16 )
            goto LABEL_25;
        }
        v14 = (__int64 **)v28[--v29];
        *v16 = v14;
        v15 = v29;
        ++v27;
        v17 = v30;
      }
    }
    else
    {
      v18 = sub_C8CA60((__int64)&v27, (__int64)&unk_4F8F810);
      if ( v18 )
      {
        *v18 = -2;
        ++v27;
        v15 = v29;
        v17 = ++v30;
      }
      else
      {
        v15 = v29;
        v17 = v30;
      }
    }
    if ( v17 == (_DWORD)v15 )
    {
      if ( v25 )
      {
        v19 = v22;
        v20 = (__int64 **)&v22[v24];
        v15 = v24;
        v14 = (__int64 **)v22;
        if ( v22 != (void **)v20 )
        {
          while ( *v14 != &qword_4F82400 )
          {
            if ( v20 == ++v14 )
            {
LABEL_20:
              while ( *v19 != &unk_4F8F810 )
              {
                if ( ++v19 == (void **)v14 )
                  goto LABEL_22;
              }
              goto LABEL_11;
            }
          }
          goto LABEL_11;
        }
        goto LABEL_22;
      }
      if ( sub_C8CA60((__int64)&v21, (__int64)&qword_4F82400) )
        goto LABEL_11;
    }
    if ( !v25 )
      goto LABEL_24;
    v19 = v22;
    v15 = v24;
    v14 = (__int64 **)&v22[v24];
    if ( v14 != (__int64 **)v22 )
      goto LABEL_20;
LABEL_22:
    if ( (unsigned int)v15 < v23 )
    {
      v24 = v15 + 1;
      *v14 = (__int64 *)&unk_4F8F810;
      ++v21;
      goto LABEL_11;
    }
LABEL_24:
    sub_C8CC70((__int64)&v21, (__int64)&unk_4F8F810, (__int64)v14, v15, v12, v13);
  }
LABEL_11:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v26, (__int64)&v21);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v32, (__int64)&v27);
  if ( !v31 )
    _libc_free((unsigned __int64)v28);
  if ( !v25 )
    _libc_free((unsigned __int64)v22);
  return a1;
}
