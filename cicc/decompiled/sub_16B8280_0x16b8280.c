// Function: sub_16B8280
// Address: 0x16b8280
//
const void *__fastcall sub_16B8280(__int64 a1, const void *a2, size_t a3)
{
  __int64 v6; // rax
  __int64 *v7; // rbx
  __int64 *i; // r15
  __int64 v9; // r13
  __int64 v10; // r14
  unsigned int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // r11
  __int64 v14; // rax
  unsigned int v15; // r10d
  _QWORD *v16; // r11
  _QWORD *v17; // r9
  void *v18; // rdi
  int v19; // eax
  __int64 v20; // rdx
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // r13
  __int64 v23; // rax
  void *v24; // rax
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rbx
  int v28; // eax
  __int64 v29; // rdx
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // r15
  __int64 v32; // [rsp+0h] [rbp-90h]
  unsigned __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 *v34; // [rsp+10h] [rbp-80h]
  _QWORD *v35; // [rsp+18h] [rbp-78h]
  _QWORD *v36; // [rsp+18h] [rbp-78h]
  _QWORD *v37; // [rsp+18h] [rbp-78h]
  unsigned int v38; // [rsp+24h] [rbp-6Ch]
  unsigned int v39; // [rsp+24h] [rbp-6Ch]
  unsigned int v40; // [rsp+24h] [rbp-6Ch]
  _QWORD *v41; // [rsp+28h] [rbp-68h]
  _QWORD *v42; // [rsp+28h] [rbp-68h]
  size_t n; // [rsp+38h] [rbp-58h]
  _QWORD v44[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v45[8]; // [rsp+50h] [rbp-40h] BYREF

  if ( !*(_BYTE *)(a1 + 152) )
    goto LABEL_2;
  v32 = sub_16B0440(a1, a2);
  v6 = *(unsigned int *)(a1 + 108);
  if ( (_DWORD)v6 == *(_DWORD *)(a1 + 112) )
  {
    v25 = sub_16B4B80((__int64)&unk_4FA0190);
    v26 = v25 + 128;
    v44[1] = a3;
    v27 = v25;
    v44[0] = a2;
    v45[0] = a1;
    sub_16B8110(v25 + 128, a2, a3, v45);
    if ( !(_BYTE)v12 )
LABEL_13:
      sub_16B0F40(v32, a1, v12);
    v28 = sub_16D1B30(v26, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32));
    if ( v28 != -1 )
    {
      v29 = *(_QWORD *)(v27 + 128);
      v30 = (unsigned __int64 *)(v29 + 8LL * v28);
      if ( v30 != (unsigned __int64 *)(v29 + 8LL * *(unsigned int *)(v27 + 136)) )
      {
        v31 = *v30;
        sub_16D1CB0(v26, *v30);
        _libc_free(v31);
      }
    }
  }
  else
  {
    v7 = *(__int64 **)(a1 + 96);
    if ( v7 != *(__int64 **)(a1 + 88) )
      v6 = *(unsigned int *)(a1 + 104);
    for ( i = &v7[v6]; i != v7; ++v7 )
    {
      if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
        break;
    }
    sub_16B55A0(v44, (__int64 *)(a1 + 80));
    v33 = a3 + 1;
    v34 = (__int64 *)v44[0];
    if ( (__int64 *)v44[0] != v7 )
    {
      n = a3;
      while ( 1 )
      {
        v9 = *v7;
        v10 = *v7 + 128;
        v11 = sub_16D19C0(v10, a2, n);
        v12 = v11;
        v13 = (_QWORD *)(*(_QWORD *)(v9 + 128) + 8LL * v11);
        if ( *v13 )
        {
          if ( *v13 != -8 )
            goto LABEL_13;
          --*(_DWORD *)(v9 + 144);
        }
        v35 = v13;
        v38 = v11;
        v14 = malloc(n + 17);
        v15 = v38;
        v16 = v35;
        v17 = (_QWORD *)v14;
        if ( v14 )
          goto LABEL_16;
        if ( n != -17 )
          break;
        v23 = malloc(1u);
        v17 = 0;
        v15 = v38;
        v16 = v35;
        if ( !v23 )
          break;
        v18 = (void *)(v23 + 16);
        v17 = (_QWORD *)v23;
LABEL_27:
        v36 = v16;
        v39 = v15;
        v41 = v17;
        v24 = memcpy(v18, a2, n);
        v16 = v36;
        v15 = v39;
        v17 = v41;
        v18 = v24;
LABEL_17:
        *((_BYTE *)v18 + n) = 0;
        *v17 = n;
        v17[1] = a1;
        *v16 = v17;
        ++*(_DWORD *)(v9 + 140);
        sub_16D1CD0(v10, v15);
        v19 = sub_16D1B30(v10, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32));
        if ( v19 != -1 )
        {
          v20 = *(_QWORD *)(v9 + 128);
          v21 = (unsigned __int64 *)(v20 + 8LL * v19);
          if ( v21 != (unsigned __int64 *)(v20 + 8LL * *(unsigned int *)(v9 + 136)) )
          {
            v22 = *v21;
            sub_16D1CB0(v10, *v21);
            _libc_free(v22);
          }
        }
        for ( ++v7; i != v7; ++v7 )
        {
          if ( (unsigned __int64)*v7 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
        if ( v34 == v7 )
        {
          a3 = n;
          goto LABEL_2;
        }
      }
      v37 = v16;
      v40 = v15;
      v42 = v17;
      sub_16BD1C0("Allocation failed");
      v17 = v42;
      v15 = v40;
      v16 = v37;
LABEL_16:
      v18 = v17 + 2;
      if ( v33 <= 1 )
        goto LABEL_17;
      goto LABEL_27;
    }
  }
LABEL_2:
  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)(a1 + 24) = a2;
  return a2;
}
