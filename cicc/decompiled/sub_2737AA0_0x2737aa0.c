// Function: sub_2737AA0
// Address: 0x2737aa0
//
__int64 __fastcall sub_2737AA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  int v14; // r8d
  unsigned int i; // eax
  __int64 v16; // rcx
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 *v19; // rax
  __int64 v20; // r9
  void *v21; // rsi
  __int64 v23; // [rsp+8h] [rbp-D8h]
  __int64 v24; // [rsp+10h] [rbp-D0h]
  __int64 *v25; // [rsp+18h] [rbp-C8h]
  __int64 v26; // [rsp+20h] [rbp-C0h] BYREF
  unsigned __int64 v27; // [rsp+28h] [rbp-B8h]
  __int64 v28; // [rsp+30h] [rbp-B0h] BYREF
  unsigned int v29; // [rsp+38h] [rbp-A8h]
  char v30; // [rsp+3Ch] [rbp-A4h]
  _QWORD v31[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v33; // [rsp+58h] [rbp-88h]
  __int64 v34; // [rsp+60h] [rbp-80h]
  int v35; // [rsp+68h] [rbp-78h]
  char v36; // [rsp+6Ch] [rbp-74h]
  _BYTE v37[64]; // [rsp+70h] [rbp-70h] BYREF
  char v38; // [rsp+B0h] [rbp-30h] BYREF

  v8 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v25 = 0;
  v24 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  if ( (_BYTE)qword_4FFA008 )
    v25 = (__int64 *)(sub_BC1CD0(a4, &unk_4F8D9A8, a3) + 8);
  v9 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v10 = *(_QWORD *)(a3 + 40);
  v11 = *(_QWORD *)(v9 + 8);
  v12 = *(unsigned int *)(v11 + 88);
  v13 = *(_QWORD *)(v11 + 72);
  if ( !(_DWORD)v12 )
    goto LABEL_25;
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
      goto LABEL_25;
    v17 = v14 + i;
    ++v14;
  }
  if ( v16 == v13 + 24 * v12 )
  {
LABEL_25:
    v18 = 0;
  }
  else
  {
    v18 = *(_QWORD *)(*(_QWORD *)(v16 + 16) + 24LL);
    if ( v18 )
    {
      v27 = 1;
      v18 += 8;
      v19 = &v28;
      do
      {
        *v19 = -4096;
        v19 += 2;
      }
      while ( v19 != (__int64 *)&v38 );
      if ( (v27 & 1) == 0 )
      {
        v23 = v18;
        sub_C7D6A0(v28, 16LL * v29, 8);
        v18 = v23;
      }
    }
  }
  v20 = *(_QWORD *)(a3 + 80);
  if ( v20 )
    v20 -= 24;
  v21 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2737780(a2, a3, v24, v8, v25, v20, v18) )
  {
    v36 = 1;
    v28 = 0x100000002LL;
    v27 = (unsigned __int64)v31;
    v31[0] = &unk_4F82408;
    v29 = 0;
    v30 = 1;
    v32 = 0;
    v33 = v37;
    v34 = 2;
    v35 = 0;
    v26 = 1;
    sub_C8CF70(a1, v21, 2, (__int64)v31, (__int64)&v26);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v37, (__int64)&v32);
    if ( !v36 )
      _libc_free((unsigned __int64)v33);
    if ( !v30 )
      _libc_free(v27);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v21;
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
