// Function: sub_30D08B0
// Address: 0x30d08b0
//
__int64 __fastcall sub_30D08B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdx
  int v13; // r10d
  unsigned int i; // eax
  __int64 v15; // r8
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // rax
  bool v22; // zf
  __int64 v23; // rdx
  int v24; // eax
  __int64 v26; // [rsp+8h] [rbp-E8h]
  __int64 *v27; // [rsp+8h] [rbp-E8h]
  __int64 v28; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v30; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v32; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+38h] [rbp-B8h]
  __int64 v34; // [rsp+40h] [rbp-B0h] BYREF
  __int64 *v35; // [rsp+48h] [rbp-A8h]
  __int64 *v36; // [rsp+50h] [rbp-A0h]
  __int64 *v37; // [rsp+58h] [rbp-98h]
  __int64 *v38; // [rsp+60h] [rbp-90h]
  __int64 *v39; // [rsp+68h] [rbp-88h]
  char v40; // [rsp+C0h] [rbp-30h] BYREF

  v8 = sub_B491C0(a2);
  v9 = *(_QWORD *)(sub_BC1CD0(a3, &unk_4F82410, v8) + 8);
  v10 = *(_QWORD *)(v9 + 72);
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 72LL) + 40LL);
  v12 = *(unsigned int *)(v9 + 88);
  if ( !(_DWORD)v12 )
    goto LABEL_22;
  v13 = 1;
  for ( i = (v12 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; i = (v12 - 1) & v16 )
  {
    v15 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v15 == &unk_4F87C68 && v11 == *(_QWORD *)(v15 + 8) )
      break;
    if ( *(_QWORD *)v15 == -4096 && *(_QWORD *)(v15 + 8) == -4096 )
      goto LABEL_22;
    v16 = v13 + i;
    ++v13;
  }
  if ( v15 == v10 + 24 * v12 )
  {
LABEL_22:
    v17 = 0;
  }
  else
  {
    v17 = *(_QWORD *)(*(_QWORD *)(v15 + 16) + 24LL);
    if ( v17 )
    {
      v33 = 1;
      v17 += 8;
      v18 = &v34;
      do
      {
        *v18 = -4096;
        v18 += 2;
      }
      while ( v18 != (__int64 *)&v40 );
      if ( (v33 & 1) == 0 )
      {
        v26 = v17;
        sub_C7D6A0(v34, 16LL * (unsigned int)v35, 8);
        v17 = v26;
      }
    }
  }
  v28 = v17;
  v19 = sub_BC1CD0(a3, &unk_4F8FAE8, v8);
  v20 = *(_QWORD *)(a2 - 32);
  v29 = a3;
  v30 = a3;
  v31 = a3;
  if ( v20 )
  {
    if ( *(_BYTE *)v20 )
    {
      v20 = 0;
    }
    else if ( *(_QWORD *)(v20 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v20 = 0;
    }
  }
  v27 = (__int64 *)(v19 + 8);
  v21 = sub_BC1CD0(a3, &unk_4F89C30, v20);
  v22 = *(_BYTE *)(a4 + 63) == 0;
  v32 = v20;
  v23 = v21 + 8;
  v33 = a4;
  v35 = &v29;
  v36 = &v31;
  v37 = &v30;
  v38 = &v28;
  v24 = (unsigned __int8)qword_502F988;
  v34 = v23;
  v39 = v27;
  if ( !v22 )
    v24 = *(unsigned __int8 *)(a4 + 62);
  sub_30CEA20(
    a1,
    a2,
    v23,
    (void (__fastcall *)(__int64 **, __int64))sub_30CA2E0,
    (__int64)&v32,
    v27,
    (int *)(a4 + 68),
    v24);
  return a1;
}
