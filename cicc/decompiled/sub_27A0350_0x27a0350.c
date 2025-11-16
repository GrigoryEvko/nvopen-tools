// Function: sub_27A0350
// Address: 0x27a0350
//
__int64 __fastcall sub_27A0350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  int v10; // r11d
  unsigned int i; // eax
  __int64 v12; // rcx
  unsigned int v13; // eax
  __int64 v14; // rbx
  __int64 *v15; // rbx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-C0h]
  __int64 v34; // [rsp+8h] [rbp-B8h]
  __int64 v35; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+20h] [rbp-A0h]
  __int64 v38; // [rsp+28h] [rbp-98h]
  __int64 v39; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v40; // [rsp+38h] [rbp-88h]
  __int64 v41; // [rsp+40h] [rbp-80h]
  int v42; // [rsp+48h] [rbp-78h]
  char v43; // [rsp+4Ch] [rbp-74h]
  _BYTE v44[16]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v45; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v46; // [rsp+68h] [rbp-58h]
  __int64 v47; // [rsp+70h] [rbp-50h]
  int v48; // [rsp+78h] [rbp-48h]
  char v49; // [rsp+7Ch] [rbp-44h]
  _BYTE v50[64]; // [rsp+80h] [rbp-40h] BYREF

  v34 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v35 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v36 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v37 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v38 = 0;
  if ( (unsigned __int8)sub_278A9A0(a2) )
    v38 = sub_BC1CD0(a4, &unk_4F8EE60, a3) + 8;
  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v8 = *(unsigned int *)(a4 + 88);
  v9 = *(_QWORD *)(a4 + 72);
  v33 = v7 + 8;
  if ( (_DWORD)v8 )
  {
    v10 = 1;
    for ( i = (v8 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F8F810 >> 9) ^ ((unsigned int)&unk_4F8F810 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v8 - 1) & v13 )
    {
      v12 = v9 + 24LL * i;
      if ( *(_UNKNOWN **)v12 == &unk_4F8F810 && a3 == *(_QWORD *)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == -4096 )
        goto LABEL_23;
      v13 = v10 + i;
      ++v10;
    }
    if ( v12 != v9 + 24 * v8 )
    {
      v14 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL);
      if ( v14 )
      {
        v15 = (__int64 *)(v14 + 8);
        sub_278A9C0(a2);
LABEL_12:
        v16 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
        *(_BYTE *)(a2 + 128) = 1;
        v17 = *v15;
        v18 = v16 + 8;
        goto LABEL_13;
      }
    }
  }
LABEL_23:
  if ( (unsigned __int8)sub_278A9C0(a2) )
  {
    v15 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
    goto LABEL_12;
  }
  v15 = 0;
  v32 = sub_BC1CD0(a4, &unk_4F8FAE8, a3);
  v17 = 0;
  *(_BYTE *)(a2 + 128) = 1;
  v18 = v32 + 8;
LABEL_13:
  if ( (unsigned __int8)sub_279FBF0(a2, a3, v34, v35, v36, v37, v38, v33, v18, v17) )
  {
    v43 = 1;
    v40 = v44;
    v39 = 0;
    v41 = 2;
    v42 = 0;
    v45 = 0;
    v46 = v50;
    v47 = 2;
    v48 = 0;
    v49 = 1;
    sub_2789D90((__int64)&v39, (__int64)&unk_4F81450, v19, v20, v21, v22);
    sub_2789D90((__int64)&v39, (__int64)&unk_4F6D3F8, v24, v25, v26, v27);
    if ( v15 )
      sub_2789D90((__int64)&v39, (__int64)&unk_4F8F810, v28, v29, v30, v31);
    sub_2789D90((__int64)&v39, (__int64)&unk_4F875F0, v28, v29, v30, v31);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v44, (__int64)&v39);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v50, (__int64)&v45);
    if ( !v49 )
      _libc_free((unsigned __int64)v46);
    if ( !v43 )
      _libc_free((unsigned __int64)v40);
  }
  else
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
  }
  return a1;
}
