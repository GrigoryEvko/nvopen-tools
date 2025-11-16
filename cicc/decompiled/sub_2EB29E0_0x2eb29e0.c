// Function: sub_2EB29E0
// Address: 0x2eb29e0
//
__int64 __fastcall sub_2EB29E0(__int64 a1, __int64 **a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  void **v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rcx
  _QWORD *v11; // rbx
  __int64 v12; // r12
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  _QWORD *v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rsi
  char v18; // al
  __int64 v19; // rcx
  void **v20; // rax
  void **v22; // rsi
  __int64 *v24; // [rsp+10h] [rbp-C0h]
  _QWORD *v26; // [rsp+20h] [rbp-B0h]
  __int64 *v27; // [rsp+28h] [rbp-A8h]
  __int64 v28; // [rsp+30h] [rbp-A0h] BYREF
  _QWORD *v29; // [rsp+38h] [rbp-98h] BYREF
  char v30[8]; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v31; // [rsp+48h] [rbp-88h]
  char v32; // [rsp+5Ch] [rbp-74h]
  unsigned __int64 v33; // [rsp+78h] [rbp-58h]
  char v34; // [rsp+8Ch] [rbp-44h]

  v5 = *(_QWORD *)(sub_2EB2140(a4, &qword_4F8A320, a3) + 8);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 2;
  v28 = v5;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  v24 = a2[1];
  if ( *a2 != v24 )
  {
    v27 = *a2;
    do
    {
      if ( (unsigned __int8)sub_2EB0040(&v28, *v27, a3) )
      {
        (*(void (__fastcall **)(char *, __int64, __int64, __int64))(*(_QWORD *)*v27 + 16LL))(v30, *v27, a3, a4);
        sub_2EB0C30(a4, a3, (__int64)v30);
        if ( v28 )
        {
          v11 = *(_QWORD **)(v28 + 432);
          v26 = &v11[4 * *(unsigned int *)(v28 + 440)];
          if ( v11 != v26 )
          {
            v12 = *v27;
            do
            {
              v29 = 0;
              v13 = (_QWORD *)sub_22077B0(0x10u);
              if ( v13 )
              {
                v13[1] = a3;
                *v13 = &unk_4A29888;
              }
              v14 = v29;
              v29 = v13;
              if ( v14 )
                (*(void (__fastcall **)(_QWORD *))(*v14 + 8LL))(v14);
              v15 = v11;
              v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 32LL))(v12);
              if ( (v11[3] & 2) == 0 )
                v15 = (_QWORD *)*v11;
              (*(void (__fastcall **)(_QWORD *, __int64, __int64, _QWORD **, char *))(v11[3] & 0xFFFFFFFFFFFFFFF8LL))(
                v15,
                v17,
                v16,
                &v29,
                v30);
              if ( v29 )
                (*(void (__fastcall **)(_QWORD *))(*v29 + 8LL))(v29);
              v11 += 4;
            }
            while ( v26 != v11 );
          }
        }
        sub_BBADB0(a1, (__int64)v30, v9, v10);
        if ( !v34 )
          _libc_free(v33);
        if ( !v32 )
          _libc_free(v31);
      }
      ++v27;
    }
    while ( v24 != v27 );
    v18 = *(_BYTE *)(a1 + 28);
    v19 = *(unsigned int *)(a1 + 72);
    if ( *(_DWORD *)(a1 + 68) != (_DWORD)v19 )
    {
LABEL_22:
      if ( !v18 )
      {
LABEL_28:
        sub_C8CC70(a1, (__int64)&unk_4FDC268, (__int64)v8, v19, v6, v7);
        return a1;
      }
      v20 = *(void ***)(a1 + 8);
      v19 = *(unsigned int *)(a1 + 20);
      v8 = &v20[v19];
      if ( v20 != v8 )
        goto LABEL_26;
LABEL_29:
      if ( *(_DWORD *)(a1 + 16) > (unsigned int)v19 )
      {
        *(_DWORD *)(a1 + 20) = v19 + 1;
        *v8 = &unk_4FDC268;
        ++*(_QWORD *)a1;
        return a1;
      }
      goto LABEL_28;
    }
    if ( !v18 )
    {
      if ( sub_C8CA60(a1, (__int64)&qword_4F82400) )
        return a1;
      v18 = *(_BYTE *)(a1 + 28);
      goto LABEL_22;
    }
  }
  v20 = *(void ***)(a1 + 8);
  v22 = &v20[*(unsigned int *)(a1 + 20)];
  v19 = *(unsigned int *)(a1 + 20);
  v8 = v20;
  if ( v20 == v22 )
    goto LABEL_29;
  while ( *v8 != &qword_4F82400 )
  {
    if ( v22 == ++v8 )
    {
LABEL_26:
      while ( *v20 != &unk_4FDC268 )
      {
        if ( v8 == ++v20 )
          goto LABEL_29;
      }
      return a1;
    }
  }
  return a1;
}
