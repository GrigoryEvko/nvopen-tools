// Function: sub_283CE90
// Address: 0x283ce90
//
__int64 __fastcall sub_283CE90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  int v13; // r8d
  unsigned int i; // eax
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 *v18; // r12
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-F0h]
  __int64 v22; // [rsp+8h] [rbp-E8h]
  __int64 v23; // [rsp+10h] [rbp-E0h]
  __int64 v24; // [rsp+18h] [rbp-D8h]
  void *v25; // [rsp+20h] [rbp-D0h]
  void *v26; // [rsp+28h] [rbp-C8h]
  __int64 v27; // [rsp+30h] [rbp-C0h] BYREF
  unsigned __int64 v28; // [rsp+38h] [rbp-B8h]
  __int64 v29; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-A8h]
  char v31; // [rsp+4Ch] [rbp-A4h]
  _QWORD v32[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v33; // [rsp+60h] [rbp-90h] BYREF
  _BYTE *v34; // [rsp+68h] [rbp-88h]
  __int64 v35; // [rsp+70h] [rbp-80h]
  int v36; // [rsp+78h] [rbp-78h]
  char v37; // [rsp+7Ch] [rbp-74h]
  _BYTE v38[64]; // [rsp+80h] [rbp-70h] BYREF
  char v39; // [rsp+C0h] [rbp-30h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v26 = (void *)(a1 + 32);
  v25 = (void *)(a1 + 80);
  if ( *(_QWORD *)(v7 + 40) == *(_QWORD *)(v7 + 48) )
    goto LABEL_7;
  v21 = sub_BC1CD0(a4, &unk_4F881D0, a3) + 8;
  v22 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v23 = sub_BC1CD0(a4, &unk_4F86630, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v9 = *(_QWORD *)(a3 + 40);
  v10 = *(_QWORD *)(v8 + 8);
  v11 = *(unsigned int *)(v10 + 88);
  v12 = *(_QWORD *)(v10 + 72);
  if ( !(_DWORD)v11 )
    goto LABEL_27;
  v13 = 1;
  for ( i = (v11 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = (v11 - 1) & v16 )
  {
    v15 = v12 + 24LL * i;
    if ( *(_UNKNOWN **)v15 == &unk_4F87C68 && v9 == *(_QWORD *)(v15 + 8) )
      break;
    if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
      goto LABEL_27;
    v16 = v13 + i;
    ++v13;
  }
  if ( v15 == v12 + 24 * v11 )
  {
LABEL_27:
    v24 = 0;
    v18 = 0;
  }
  else
  {
    v18 = *(__int64 **)(*(_QWORD *)(v15 + 16) + 24LL);
    if ( v18 )
    {
      v28 = 1;
      v24 = (__int64)(v18 + 1);
      v19 = &v29;
      do
      {
        *v19 = -4096;
        v19 += 2;
      }
      while ( v19 != (__int64 *)&v39 );
      if ( (v28 & 1) == 0 )
        sub_C7D6A0(v29, 16LL * v30, 8);
      v18 = (__int64 *)v18[2];
      if ( v18 )
        v18 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8);
    }
    else
    {
      v24 = 0;
    }
  }
  v20 = sub_BC1CD0(a4, &unk_4F86D28, a3);
  if ( !(unsigned __int8)sub_2839ED0(v7 + 8, v22, v18, v24, v21, v23, v20 + 8) )
  {
LABEL_7:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_QWORD *)(a1 + 8) = v26;
    *(_DWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 56) = v25;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v29 = 0x100000002LL;
    v28 = (unsigned __int64)v32;
    v32[0] = &unk_4F81450;
    v37 = 1;
    v30 = 0;
    v31 = 1;
    v33 = 0;
    v34 = v38;
    v35 = 2;
    v36 = 0;
    v27 = 1;
    if ( &unk_4F81450 != (_UNKNOWN *)&qword_4F82400 && &unk_4F81450 != &unk_4F875F0 )
    {
      HIDWORD(v29) = 2;
      v32[1] = &unk_4F875F0;
      v27 = 2;
    }
    sub_C8CF70(a1, v26, 2, (__int64)v32, (__int64)&v27);
    sub_C8CF70(a1 + 48, v25, 2, (__int64)v38, (__int64)&v33);
    if ( !v37 )
      _libc_free((unsigned __int64)v34);
    if ( !v31 )
      _libc_free(v28);
  }
  return a1;
}
