// Function: sub_2AF5D00
// Address: 0x2af5d00
//
__int64 __fastcall sub_2AF5D00(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdi
  int v14; // r11d
  unsigned int i; // eax
  __int64 v16; // rcx
  unsigned int v17; // eax
  _QWORD *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v22; // rbx
  __int64 v23; // rcx
  __int64 *v24; // rax
  unsigned __int16 v25; // bx
  __int64 v26; // rdi
  __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned __int16 v31; // [rsp+Eh] [rbp-E2h]
  void *v32; // [rsp+20h] [rbp-D0h]
  void *v33; // [rsp+28h] [rbp-C8h]
  _QWORD *v34; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v35; // [rsp+38h] [rbp-B8h]
  __int64 v36; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v37; // [rsp+48h] [rbp-A8h]
  char v38; // [rsp+4Ch] [rbp-A4h]
  _BYTE v39[16]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v41; // [rsp+68h] [rbp-88h]
  __int64 v42; // [rsp+70h] [rbp-80h]
  int v43; // [rsp+78h] [rbp-78h]
  char v44; // [rsp+7Ch] [rbp-74h]
  _BYTE v45[64]; // [rsp+80h] [rbp-70h] BYREF
  char v46; // [rsp+C0h] [rbp-30h] BYREF

  v8 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v33 = (void *)(a1 + 32);
  v32 = (void *)(a1 + 80);
  a2[2] = v8 + 8;
  if ( *(_QWORD *)(v8 + 40) == *(_QWORD *)(v8 + 48) )
    goto LABEL_7;
  a2[1] = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  a2[3] = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  a2[4] = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  a2[6] = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  a2[8] = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  a2[7] = sub_BC1CD0(a4, &unk_4F86B68, a3) + 8;
  a2[10] = sub_BC1CD0(a4, &unk_4F8FAE8, a3) + 8;
  a2[9] = sub_BC1CD0(a4, &unk_4F86D28, a3) + 8;
  v9 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v10 = *(_QWORD *)(a3 + 40);
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned int *)(v11 + 88);
  v13 = *(_QWORD *)(v11 + 72);
  if ( !(_DWORD)v12 )
    goto LABEL_38;
  v14 = 1;
  for ( i = (v12 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4)))); ; i = (v12 - 1) & v17 )
  {
    v16 = v13 + 24LL * i;
    if ( *(_UNKNOWN **)v16 == &unk_4F87C68 && v10 == *(_QWORD *)(v16 + 8) )
      break;
    if ( *(_QWORD *)v16 == -4096 && *(_QWORD *)(v16 + 8) == -4096 )
      goto LABEL_38;
    v17 = v14 + i;
    ++v14;
  }
  if ( v16 != v13 + 24 * v12 && (v22 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL)) != 0 )
  {
    v35 = 1;
    v23 = v22 + 8;
    v24 = &v36;
    do
    {
      *v24 = -4096;
      v24 += 2;
    }
    while ( v24 != (__int64 *)&v46 );
    if ( (v35 & 1) == 0 )
    {
      sub_C7D6A0(v36, 16LL * v37, 8);
      v23 = v22 + 8;
    }
    a2[11] = v23;
    a2[5] = 0;
    if ( *(_QWORD *)(v22 + 16) )
      a2[5] = sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8;
  }
  else
  {
LABEL_38:
    a2[11] = 0;
    a2[5] = 0;
  }
  v25 = sub_2AF5B40((__int64)a2);
  if ( !(_BYTE)v25 )
  {
LABEL_7:
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = v33;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v32;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    v18 = sub_AE6EC0(a1, (__int64)&qword_4F82400);
    if ( *(_BYTE *)(a1 + 28) )
      v19 = *(unsigned int *)(a1 + 20);
    else
      v19 = *(unsigned int *)(a1 + 16);
    v20 = *(_QWORD *)(a1 + 8) + 8 * v19;
    v34 = v18;
    v35 = v20;
    sub_254BBF0((__int64)&v34);
  }
  else
  {
    v26 = *(_QWORD *)(a3 + 40);
    v34 = 0;
    v35 = (unsigned __int64)v39;
    v36 = 2;
    v37 = 0;
    v38 = 1;
    v40 = 0;
    v41 = v45;
    v42 = 2;
    v43 = 0;
    v44 = 1;
    if ( (unsigned __int8)sub_AEA460(v26) && a3 + 72 != *(_QWORD *)(a3 + 80) )
    {
      v31 = v25;
      v27 = *(_QWORD *)(a3 + 80);
      do
      {
        v28 = v27 - 24;
        if ( !v27 )
          v28 = 0;
        sub_F3F2F0(v28, a3);
        v27 = *(_QWORD *)(v27 + 8);
      }
      while ( a3 + 72 != v27 );
      v25 = v31;
    }
    sub_2AAE570((__int64)&v34, (__int64)&unk_4F875F0);
    sub_2AAE570((__int64)&v34, (__int64)&unk_4F81450);
    sub_2AAE570((__int64)&v34, (__int64)&unk_4F881D0);
    sub_2AAE570((__int64)&v34, (__int64)&unk_4F86D28);
    if ( HIBYTE(v25) )
    {
      sub_BC1CD0(a4, &unk_500CD08, a3);
      sub_2AAE570((__int64)&v34, (__int64)&unk_500CD08);
    }
    else if ( HIDWORD(v42) != v43 || !(unsigned __int8)sub_B19060((__int64)&v34, (__int64)&qword_4F82400, v29, v30) )
    {
      sub_AE6EC0((__int64)&v34, (__int64)&unk_4F82408);
    }
    sub_C8CF70(a1, v33, 2, (__int64)v39, (__int64)&v34);
    sub_C8CF70(a1 + 48, v32, 2, (__int64)v45, (__int64)&v40);
    if ( !v44 )
      _libc_free((unsigned __int64)v41);
    if ( !v38 )
      _libc_free(v35);
  }
  return a1;
}
