// Function: sub_269EB30
// Address: 0x269eb30
//
__int64 __fastcall sub_269EB30(__int64 *a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  unsigned __int8 *v7; // r14
  __int64 (__fastcall *v8)(__int64); // rax
  unsigned __int8 *v9; // rdi
  __int64 (__fastcall *v10)(__int64); // rax
  unsigned int v11; // r12d
  __int64 v12; // r13
  __int64 v13; // r9
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rbx
  __int64 v18; // rax
  _DWORD *v19; // rcx
  _BYTE *v20; // rbx
  _BYTE *v21; // rbx
  unsigned __int64 v22; // r15
  _BYTE *v23; // r14
  __int64 v24; // rbx
  int v25; // ebx
  __int64 v26; // [rsp-10h] [rbp-150h]
  __int64 v27; // [rsp+8h] [rbp-138h]
  unsigned __int64 v28; // [rsp+18h] [rbp-128h] BYREF
  unsigned __int64 v29; // [rsp+20h] [rbp-120h] BYREF
  _QWORD v30[3]; // [rsp+28h] [rbp-118h] BYREF
  char v31; // [rsp+44h] [rbp-FCh]
  _BYTE v32[16]; // [rsp+48h] [rbp-F8h] BYREF
  _BYTE v33[8]; // [rsp+58h] [rbp-E8h] BYREF
  unsigned __int64 v34; // [rsp+60h] [rbp-E0h]
  char v35; // [rsp+74h] [rbp-CCh]
  _BYTE v36[32]; // [rsp+78h] [rbp-C8h] BYREF
  char v37; // [rsp+98h] [rbp-A8h]
  char v38; // [rsp+99h] [rbp-A7h]
  char v39; // [rsp+9Ah] [rbp-A6h]
  char v40; // [rsp+9Bh] [rbp-A5h]
  _BYTE v41[8]; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned __int64 v42; // [rsp+A8h] [rbp-98h]
  char v43; // [rsp+BCh] [rbp-84h]
  _BYTE v44[16]; // [rsp+C0h] [rbp-80h] BYREF
  _BYTE v45[8]; // [rsp+D0h] [rbp-70h] BYREF
  unsigned __int64 v46; // [rsp+D8h] [rbp-68h]
  char v47; // [rsp+ECh] [rbp-54h]
  _BYTE v48[80]; // [rsp+F0h] [rbp-50h] BYREF

  v3 = *a1;
  v4 = sub_B43CB0(*a2);
  v30[0] = 0;
  v29 = v4 & 0xFFFFFFFFFFFFFFFCLL;
  nullsub_1518();
  v5 = sub_269E5E0(v3, v29, 0, a1[1], 1, 0, 1);
  v6 = v26;
  if ( !v5 )
    return 0;
  v7 = (unsigned __int8 *)v5;
  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
  if ( v8 == sub_2534F50 )
  {
    v9 = v7 + 88;
    v10 = *(__int64 (__fastcall **)(__int64))(*((_QWORD *)v7 + 11) + 16LL);
    if ( v10 == sub_2505E30 )
    {
LABEL_4:
      v11 = v9[9];
      goto LABEL_5;
    }
  }
  else
  {
    v9 = (unsigned __int8 *)((__int64 (__fastcall *)(unsigned __int8 *, unsigned __int64, __int64))v8)(v7, v29, v26);
    v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL);
    if ( v10 == sub_2505E30 )
      goto LABEL_4;
  }
  v11 = ((__int64 (__fastcall *)(unsigned __int8 *, unsigned __int64, __int64))v10)(v9, v29, v6);
LABEL_5:
  if ( !(_BYTE)v11 )
    return 0;
  v12 = a1[2];
  (*(void (__fastcall **)(unsigned __int64 *, unsigned __int8 *, __int64))(*(_QWORD *)v7 + 136LL))(&v29, v7, *a2);
  v14 = *(unsigned int *)(v12 + 8);
  v15 = v14;
  if ( *(_DWORD *)(v12 + 12) <= (unsigned int)v14 )
  {
    v27 = sub_C8D7D0(v12, v12 + 16, 0, 0xF0u, &v28, v13);
    v18 = 240LL * *(unsigned int *)(v12 + 8);
    v19 = (_DWORD *)(v18 + v27);
    v20 = (_BYTE *)(v18 + v27);
    if ( v18 + v27 )
    {
      *v19 = v29;
      sub_C8CF70((__int64)(v19 + 2), v19 + 10, 2, (__int64)v32, (__int64)v30);
      sub_C8CF70((__int64)(v20 + 56), v20 + 88, 4, (__int64)v36, (__int64)v33);
      v20[120] = v37;
      v20[121] = v38;
      v20[122] = v39;
      v20[123] = v40;
      sub_C8CF70((__int64)(v20 + 128), v20 + 160, 2, (__int64)v44, (__int64)v41);
      sub_C8CF70((__int64)(v20 + 176), v20 + 208, 4, (__int64)v48, (__int64)v45);
      v18 = 240LL * *(unsigned int *)(v12 + 8);
    }
    v21 = *(_BYTE **)v12;
    v22 = *(_QWORD *)v12 + v18;
    if ( *(_QWORD *)v12 == v22 )
      goto LABEL_39;
    v23 = (_BYTE *)v27;
    do
    {
      if ( v23 )
      {
        *v23 = *v21;
        v23[1] = v21[1];
        v23[2] = v21[2];
        v23[3] = v21[3];
        sub_C8CF70((__int64)(v23 + 8), v23 + 40, 2, (__int64)(v21 + 40), (__int64)(v21 + 8));
        sub_C8CF70((__int64)(v23 + 56), v23 + 88, 4, (__int64)(v21 + 88), (__int64)(v21 + 56));
        v23[120] = v21[120];
        v23[121] = v21[121];
        v23[122] = v21[122];
        v23[123] = v21[123];
        sub_C8CF70((__int64)(v23 + 128), v23 + 160, 2, (__int64)(v21 + 160), (__int64)(v21 + 128));
        sub_C8CF70((__int64)(v23 + 176), v23 + 208, 4, (__int64)(v21 + 208), (__int64)(v21 + 176));
      }
      v21 += 240;
      v23 += 240;
    }
    while ( (_BYTE *)v22 != v21 );
    v22 = *(_QWORD *)v12;
    v24 = *(_QWORD *)v12 + 240LL * *(unsigned int *)(v12 + 8);
    if ( v24 == *(_QWORD *)v12 )
    {
LABEL_39:
      v25 = v28;
      if ( v12 + 16 != v22 )
        _libc_free(v22);
      ++*(_DWORD *)(v12 + 8);
      *(_DWORD *)(v12 + 12) = v25;
      *(_QWORD *)v12 = v27;
      goto LABEL_10;
    }
    while ( 1 )
    {
      v24 -= 240;
      if ( *(_BYTE *)(v24 + 204) )
      {
        if ( *(_BYTE *)(v24 + 156) )
          goto LABEL_30;
      }
      else
      {
        _libc_free(*(_QWORD *)(v24 + 184));
        if ( *(_BYTE *)(v24 + 156) )
        {
LABEL_30:
          if ( *(_BYTE *)(v24 + 84) )
            goto LABEL_31;
          goto LABEL_36;
        }
      }
      _libc_free(*(_QWORD *)(v24 + 136));
      if ( *(_BYTE *)(v24 + 84) )
      {
LABEL_31:
        if ( !*(_BYTE *)(v24 + 36) )
          goto LABEL_37;
        goto LABEL_32;
      }
LABEL_36:
      _libc_free(*(_QWORD *)(v24 + 64));
      if ( !*(_BYTE *)(v24 + 36) )
LABEL_37:
        _libc_free(*(_QWORD *)(v24 + 16));
LABEL_32:
      if ( v24 == v22 )
      {
        v22 = *(_QWORD *)v12;
        goto LABEL_39;
      }
    }
  }
  v16 = *(_QWORD *)v12 + 240 * v14;
  if ( v16 )
  {
    *(_DWORD *)v16 = v29;
    sub_C8CF70(v16 + 8, (void *)(v16 + 40), 2, (__int64)v32, (__int64)v30);
    sub_C8CF70(v16 + 56, (void *)(v16 + 88), 4, (__int64)v36, (__int64)v33);
    *(_BYTE *)(v16 + 120) = v37;
    *(_BYTE *)(v16 + 121) = v38;
    *(_BYTE *)(v16 + 122) = v39;
    *(_BYTE *)(v16 + 123) = v40;
    sub_C8CF70(v16 + 128, (void *)(v16 + 160), 2, (__int64)v44, (__int64)v41);
    sub_C8CF70(v16 + 176, (void *)(v16 + 208), 4, (__int64)v48, (__int64)v45);
    v15 = *(_DWORD *)(v12 + 8);
  }
  *(_DWORD *)(v12 + 8) = v15 + 1;
LABEL_10:
  if ( !v47 )
    _libc_free(v46);
  if ( !v43 )
    _libc_free(v42);
  if ( !v35 )
    _libc_free(v34);
  if ( !v31 )
    _libc_free(v30[1]);
  return v11;
}
