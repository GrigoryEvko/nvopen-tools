// Function: sub_22C5240
// Address: 0x22c5240
//
__int64 __fastcall sub_22C5240(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  bool v9; // dl
  unsigned int v11; // eax
  int v12; // eax
  unsigned int v13; // esi
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rax
  char v17; // dl
  __int64 v18; // rdi
  int v19; // r8d
  unsigned int v20; // eax
  __int64 v21; // r9
  __int64 v22; // rsi
  unsigned int v23; // esi
  unsigned int v24; // eax
  int v25; // ecx
  unsigned int v26; // edi
  _QWORD *v27; // r15
  int v28; // r11d
  _QWORD *v29; // r10
  _QWORD *v30; // [rsp+8h] [rbp-A8h]
  _QWORD *v31; // [rsp+10h] [rbp-A0h] BYREF
  _QWORD *v32; // [rsp+18h] [rbp-98h] BYREF
  __int64 v33; // [rsp+20h] [rbp-90h] BYREF
  __int64 v34; // [rsp+28h] [rbp-88h]
  __int64 v35; // [rsp+30h] [rbp-80h]
  __int64 v36; // [rsp+40h] [rbp-70h] BYREF
  __int64 v37; // [rsp+48h] [rbp-68h]
  __int64 v38; // [rsp+50h] [rbp-60h]
  unsigned __int8 v39; // [rsp+58h] [rbp-58h] BYREF
  char v40; // [rsp+59h] [rbp-57h]
  __int64 v41; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v42; // [rsp+68h] [rbp-48h]
  __int64 v43; // [rsp+70h] [rbp-40h] BYREF
  unsigned int v44; // [rsp+78h] [rbp-38h]

  v7 = sub_22C23A0(a1, a3);
  v8 = *(_BYTE *)a4;
  v9 = a2 != -8192 && a2 != -4096 && a2 != 0;
  if ( *(_BYTE *)a4 == 6 )
  {
    v35 = a2;
    v33 = 0;
    v34 = 0;
    if ( v9 )
      sub_BD73F0((__int64)&v33);
    v17 = *(_BYTE *)(v7 + 280) & 1;
    if ( v17 )
    {
      v18 = v7 + 288;
      v19 = 3;
    }
    else
    {
      v23 = *(_DWORD *)(v7 + 296);
      v18 = *(_QWORD *)(v7 + 288);
      v19 = v23 - 1;
      if ( !v23 )
      {
        v32 = 0;
        v24 = *(_DWORD *)(v7 + 280);
        ++*(_QWORD *)(v7 + 272);
        v25 = (v24 >> 1) + 1;
        goto LABEL_34;
      }
    }
    v20 = v19 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v21 = v18 + 24LL * v20;
    v22 = *(_QWORD *)(v21 + 16);
    if ( v35 == v22 )
    {
LABEL_31:
      sub_D68D70(&v33);
      return sub_22BECA0(a1, a2);
    }
    v28 = 1;
    v29 = 0;
    while ( v22 != -4096 )
    {
      if ( v29 || v22 != -8192 )
        v21 = (__int64)v29;
      v20 = v19 & (v28 + v20);
      v22 = *(_QWORD *)(v18 + 24LL * v20 + 16);
      if ( v35 == v22 )
        goto LABEL_31;
      ++v28;
      v29 = (_QWORD *)v21;
      v21 = v18 + 24LL * v20;
    }
    if ( !v29 )
      v29 = (_QWORD *)v21;
    v32 = v29;
    v24 = *(_DWORD *)(v7 + 280);
    ++*(_QWORD *)(v7 + 272);
    v25 = (v24 >> 1) + 1;
    if ( v17 )
    {
      v26 = 12;
      v23 = 4;
LABEL_35:
      if ( v26 <= 4 * v25 )
      {
        v23 *= 2;
      }
      else if ( v23 - *(_DWORD *)(v7 + 284) - v25 > v23 >> 3 )
      {
LABEL_37:
        *(_DWORD *)(v7 + 280) = (2 * (v24 >> 1) + 2) | v24 & 1;
        v27 = v32;
        v36 = 0;
        v37 = 0;
        v38 = -4096;
        if ( v32[2] != -4096 )
          --*(_DWORD *)(v7 + 284);
        sub_D68D70(&v36);
        sub_22BDC40(v27, v35);
        goto LABEL_31;
      }
      sub_22C4510(v7 + 272, v23);
      sub_22C3870(v7 + 272, (__int64)&v33, &v32);
      v24 = *(_DWORD *)(v7 + 280);
      goto LABEL_37;
    }
    v23 = *(_DWORD *)(v7 + 296);
LABEL_34:
    v26 = 3 * v23;
    goto LABEL_35;
  }
  v36 = 0;
  v37 = 0;
  v38 = a2;
  if ( v9 )
  {
    sub_BD73F0((__int64)&v36);
    v8 = *(_BYTE *)a4;
    v40 = 0;
    v39 = v8;
    if ( v8 > 3u )
    {
LABEL_4:
      if ( (unsigned __int8)(v8 - 4) <= 1u )
      {
        v42 = *(_DWORD *)(a4 + 16);
        if ( v42 > 0x40 )
          sub_C43780((__int64)&v41, (const void **)(a4 + 8));
        else
          v41 = *(_QWORD *)(a4 + 8);
        v44 = *(_DWORD *)(a4 + 32);
        if ( v44 > 0x40 )
          sub_C43780((__int64)&v43, (const void **)(a4 + 24));
        else
          v43 = *(_QWORD *)(a4 + 24);
        v40 = *(_BYTE *)(a4 + 1);
      }
      goto LABEL_10;
    }
  }
  else
  {
    v39 = v8;
    v40 = 0;
    if ( v8 > 3u )
      goto LABEL_4;
  }
  if ( v8 > 1u )
    v41 = *(_QWORD *)(a4 + 8);
LABEL_10:
  if ( !(unsigned __int8)sub_22C3940(v7, (__int64)&v36, &v31) )
  {
    v32 = v31;
    v11 = *(_DWORD *)(v7 + 8);
    ++*(_QWORD *)v7;
    v12 = (v11 >> 1) + 1;
    if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
    {
      v14 = 12;
      v13 = 4;
    }
    else
    {
      v13 = *(_DWORD *)(v7 + 24);
      v14 = 3 * v13;
    }
    v15 = (unsigned int)(4 * v12);
    if ( (unsigned int)v14 <= (unsigned int)v15 )
    {
      v13 *= 2;
    }
    else
    {
      v14 = v13 - (v12 + *(_DWORD *)(v7 + 12));
      v15 = v13 >> 3;
      if ( (unsigned int)v14 > (unsigned int)v15 )
      {
LABEL_23:
        *(_DWORD *)(v7 + 8) = *(_DWORD *)(v7 + 8) & 1 | (2 * v12);
        v16 = v32;
        v33 = 0;
        v34 = 0;
        v35 = -4096;
        if ( v32[2] != -4096 )
          --*(_DWORD *)(v7 + 12);
        v30 = v16;
        sub_D68D70(&v33);
        sub_22BDC40(v30, v38);
        sub_22C0650((__int64)(v30 + 3), &v39);
        goto LABEL_11;
      }
    }
    sub_22C4F30(v7, v13, v15, v14);
    sub_22C3940(v7, (__int64)&v36, &v32);
    v12 = (*(_DWORD *)(v7 + 8) >> 1) + 1;
    goto LABEL_23;
  }
LABEL_11:
  sub_22C0090(&v39);
  if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
    sub_BD60C0(&v36);
  return sub_22BECA0(a1, a2);
}
