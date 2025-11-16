// Function: sub_D4E470
// Address: 0xd4e470
//
__int64 __fastcall sub_D4E470(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 result; // rax
  __int64 v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // rdi
  _BYTE *v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r12
  __int64 v15; // r9
  int v16; // r11d
  __int64 *v17; // rdi
  __int64 v18; // rcx
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r8
  _DWORD *v22; // rax
  bool v23; // zf
  int v24; // eax
  int v25; // edx
  __int64 v26; // [rsp+18h] [rbp-468h] BYREF
  __int64 v27[2]; // [rsp+20h] [rbp-460h] BYREF
  __int64 *v28; // [rsp+30h] [rbp-450h] BYREF
  _BYTE *v29; // [rsp+38h] [rbp-448h] BYREF
  __int64 v30; // [rsp+40h] [rbp-440h]
  _BYTE v31[328]; // [rsp+48h] [rbp-438h] BYREF
  __int64 *v32; // [rsp+190h] [rbp-2F0h] BYREF
  __int64 *v33; // [rsp+2F0h] [rbp-190h] BYREF
  _BYTE *v34; // [rsp+2F8h] [rbp-188h] BYREF
  __int64 v35; // [rsp+300h] [rbp-180h]
  _BYTE v36[376]; // [rsp+308h] [rbp-178h] BYREF

  v2 = *a1;
  v27[0] = (__int64)a1;
  v27[1] = a2;
  v3 = **(_QWORD **)(v2 + 32);
  sub_D4E110(&v33, v3, v27);
  v29 = v31;
  v28 = v33;
  v30 = 0x800000000LL;
  if ( (_DWORD)v35 )
  {
    v3 = (__int64)&v34;
    sub_D4C550((__int64)&v29, (__int64)&v34, v4, v5, v6, v7);
  }
  if ( v34 != v36 )
    _libc_free(v34, v3);
  v33 = v27;
  v32 = v27;
  v34 = v36;
  v35 = 0x800000000LL;
LABEL_6:
  result = (unsigned int)v30;
  while ( result )
  {
    v9 = *(_QWORD *)&v29[40 * result - 8];
    v10 = v28;
    v26 = v9;
    v11 = *v28;
    v12 = *(_BYTE **)(*v28 + 48);
    if ( v12 == *(_BYTE **)(*v28 + 56) )
    {
      sub_9319A0(v11 + 40, v12, &v26);
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v9;
        v12 = *(_BYTE **)(v11 + 48);
      }
      *(_QWORD *)(v11 + 48) = v12 + 8;
    }
    v13 = *v10;
    v3 = *(unsigned int *)(v13 + 32);
    v14 = (__int64)(*(_QWORD *)(v13 + 48) - *(_QWORD *)(v13 + 40)) >> 3;
    if ( !(_DWORD)v3 )
    {
      v32 = 0;
      ++*(_QWORD *)(v13 + 8);
      goto LABEL_35;
    }
    v15 = *(_QWORD *)(v13 + 16);
    v16 = 1;
    v17 = 0;
    v18 = v26;
    v19 = (v3 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
    v20 = (_QWORD *)(v15 + 16LL * v19);
    v21 = *v20;
    if ( v26 != *v20 )
    {
      while ( v21 != -4096 )
      {
        if ( !v17 && v21 == -8192 )
          v17 = v20;
        v19 = (v3 - 1) & (v16 + v19);
        v20 = (_QWORD *)(v15 + 16LL * v19);
        v21 = *v20;
        if ( v26 == *v20 )
          goto LABEL_17;
        ++v16;
      }
      if ( !v17 )
        v17 = v20;
      v32 = v17;
      v24 = *(_DWORD *)(v13 + 24);
      ++*(_QWORD *)(v13 + 8);
      v25 = v24 + 1;
      if ( 4 * (v24 + 1) >= (unsigned int)(3 * v3) )
      {
LABEL_35:
        LODWORD(v3) = 2 * v3;
      }
      else if ( (int)v3 - *(_DWORD *)(v13 + 28) - v25 > (unsigned int)v3 >> 3 )
      {
LABEL_30:
        *(_DWORD *)(v13 + 24) = v25;
        if ( *v17 != -4096 )
          --*(_DWORD *)(v13 + 28);
        *v17 = v18;
        v22 = v17 + 1;
        *((_DWORD *)v17 + 2) = 0;
        goto LABEL_18;
      }
      sub_B23080(v13 + 8, v3);
      v3 = (__int64)&v26;
      sub_B1C700(v13 + 8, &v26, &v32);
      v18 = v26;
      v17 = v32;
      v25 = *(_DWORD *)(v13 + 24) + 1;
      goto LABEL_30;
    }
LABEL_17:
    v22 = v20 + 1;
LABEL_18:
    *v22 = v14;
    v23 = (_DWORD)v30 == 1;
    result = (unsigned int)(v30 - 1);
    LODWORD(v30) = v30 - 1;
    if ( !v23 )
    {
      sub_D4DD40(&v28);
      goto LABEL_6;
    }
  }
  if ( v29 != v31 )
    return _libc_free(v29, v3);
  return result;
}
