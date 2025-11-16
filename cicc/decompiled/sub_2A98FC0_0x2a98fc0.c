// Function: sub_2A98FC0
// Address: 0x2a98fc0
//
__int64 __fastcall sub_2A98FC0(__int64 a1, __int64 *a2)
{
  char v4; // di
  int v5; // ecx
  int v6; // edx
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // r14
  int v10; // r11d
  unsigned int i; // eax
  __int64 v12; // r9
  int v13; // r13d
  unsigned int v14; // eax
  __int64 v15; // rax
  char *v16; // r13
  int v17; // r8d
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rcx
  unsigned __int64 v22; // rsi
  int v23; // edx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  _DWORD *v27; // rsi
  bool v28; // zf
  int v30; // eax
  unsigned __int64 v31; // r12
  __int64 v32; // rdi
  char v33[4]; // [rsp+0h] [rbp-C0h] BYREF
  int v34; // [rsp+4h] [rbp-BCh]
  int v35; // [rsp+8h] [rbp-B8h]
  int v36; // [rsp+Ch] [rbp-B4h]
  __int64 v37; // [rsp+10h] [rbp-B0h] BYREF
  _BYTE *v38; // [rsp+18h] [rbp-A8h]
  __int64 v39; // [rsp+20h] [rbp-A0h]
  int v40; // [rsp+28h] [rbp-98h]
  char v41; // [rsp+2Ch] [rbp-94h]
  _BYTE v42[32]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v43; // [rsp+50h] [rbp-70h] BYREF
  int v44; // [rsp+58h] [rbp-68h]
  _BYTE v45[8]; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 v46; // [rsp+68h] [rbp-58h]
  char v47; // [rsp+7Ch] [rbp-44h]
  _BYTE v48[64]; // [rsp+80h] [rbp-40h] BYREF

  v4 = *(_BYTE *)a2;
  v5 = *((_DWORD *)a2 + 1);
  v36 = 0;
  v6 = *((_DWORD *)a2 + 2);
  v7 = *(_DWORD *)(a1 + 24);
  v33[0] = v4;
  v34 = v5;
  v35 = v6;
  if ( !v7 )
  {
    ++*(_QWORD *)a1;
    v43 = 0;
    goto LABEL_12;
  }
  v9 = 0;
  v10 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned __int64)(unsigned int)(37 * v6) << 32)
              | (unsigned int)((0xBF58476D1CE4E5B9LL
                              * ((unsigned int)(1512728442 * v4) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))) >> 31)
              ^ (-1747130070 * v4))) >> 31)
           ^ (484763065
            * (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(1512728442 * v4) | ((unsigned __int64)(unsigned int)(37 * v5) << 32))) >> 31)
             ^ (-1747130070 * v4)))); ; i = (v7 - 1) & v14 )
  {
    v8 = *(_QWORD *)(a1 + 8);
    v12 = v8 + 16LL * i;
    v13 = *(_DWORD *)(v12 + 8);
    if ( v6 == v13 && v5 == *(_DWORD *)(v12 + 4) && v4 == *(_BYTE *)v12 )
    {
      v15 = *(unsigned int *)(v12 + 12);
      return *(_QWORD *)(a1 + 32) + 80 * v15 + 16;
    }
    if ( v13 == -1 )
      break;
    if ( v13 == -2 && *(_DWORD *)(v12 + 4) == -2 && *(_BYTE *)v12 == 0xFE && !v9 )
      v9 = v8 + 16LL * i;
LABEL_7:
    v14 = v10 + i;
    ++v10;
  }
  if ( *(_DWORD *)(v12 + 4) != -1 || *(_BYTE *)v12 != 0xFF )
    goto LABEL_7;
  v30 = *(_DWORD *)(a1 + 16);
  if ( !v9 )
    v9 = v12;
  ++*(_QWORD *)a1;
  v17 = v30 + 1;
  v43 = v9;
  if ( 4 * (v30 + 1) >= 3 * v7 )
  {
LABEL_12:
    v16 = (char *)&v43;
    sub_2A98CE0(a1, 2 * v7);
    goto LABEL_13;
  }
  v16 = (char *)&v43;
  if ( v7 - *(_DWORD *)(a1 + 20) - v17 <= v7 >> 3 )
  {
    sub_2A98CE0(a1, v7);
LABEL_13:
    sub_2A92C60(a1, v33, &v43);
    v9 = v43;
    v6 = v35;
    v5 = v34;
    v4 = v33[0];
    v17 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v17;
  if ( *(_DWORD *)(v9 + 8) != -1 || *(_DWORD *)(v9 + 4) != -1 || *(_BYTE *)v9 != 0xFF )
    --*(_DWORD *)(a1 + 20);
  *(_DWORD *)(v9 + 4) = v5;
  *(_BYTE *)v9 = v4;
  *(_DWORD *)(v9 + 8) = v6;
  v37 = 0;
  *(_DWORD *)(v9 + 12) = v36;
  v18 = *a2;
  v38 = v42;
  v43 = v18;
  LODWORD(v18) = *((_DWORD *)a2 + 2);
  v39 = 4;
  v44 = v18;
  v40 = 0;
  v41 = 1;
  sub_C8CF70((__int64)v45, v48, 4, (__int64)v42, (__int64)&v37);
  v21 = *(unsigned int *)(a1 + 40);
  v22 = v21 + 1;
  v23 = v21;
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    v31 = *(_QWORD *)(a1 + 32);
    v32 = a1 + 32;
    if ( v31 > (unsigned __int64)&v43 || (unsigned __int64)&v43 >= v31 + 80 * v21 )
    {
      sub_2A92EF0(v32, v22, v21, v21, v19, v20);
      v21 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v23 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2A92EF0(v32, v22, v21, v21, v19, v20);
      v24 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v16 = (char *)&v43 + v24 - v31;
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
  }
  v25 = v24 + 80 * v21;
  if ( v25 )
  {
    v26 = v25 + 16;
    v27 = (_DWORD *)(v25 + 48);
    *((_BYTE *)v27 - 48) = *v16;
    *(v27 - 11) = *((_DWORD *)v16 + 1);
    *(v27 - 10) = *((_DWORD *)v16 + 2);
    sub_C8CF70(v26, v27, 4, (__int64)(v16 + 48), (__int64)(v16 + 16));
    v23 = *(_DWORD *)(a1 + 40);
  }
  v28 = v47 == 0;
  *(_DWORD *)(a1 + 40) = v23 + 1;
  if ( v28 )
    _libc_free(v46);
  if ( !v41 )
    _libc_free((unsigned __int64)v38);
  v15 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v9 + 12) = v15;
  return *(_QWORD *)(a1 + 32) + 80 * v15 + 16;
}
