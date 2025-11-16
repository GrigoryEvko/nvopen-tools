// Function: sub_2926390
// Address: 0x2926390
//
__int64 __fastcall sub_2926390(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r12
  char v9; // di
  __int64 v10; // rsi
  int v11; // r8d
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  __int64 v15; // rdx
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // eax
  __int64 v23; // r14
  const char *v24; // rdx
  unsigned int v25; // r11d
  __int64 v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // r10
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r12
  __int64 v38; // rbx
  __int64 v39; // r12
  __int64 v40; // rdx
  unsigned int v41; // esi
  int v42; // r11d
  unsigned int v43; // [rsp+14h] [rbp-7Ch]
  __int64 v44; // [rsp+18h] [rbp-78h]
  __int64 v46; // [rsp+20h] [rbp-70h]
  __int64 v47; // [rsp+28h] [rbp-68h] BYREF
  const char *v48[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v49; // [rsp+50h] [rbp-40h]

  v7 = a1;
  v47 = a1;
  v9 = *(_BYTE *)(a5 + 8) & 1;
  if ( v9 )
  {
    v10 = a5 + 16;
    v11 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(a5 + 24);
    v10 = *(_QWORD *)(a5 + 16);
    if ( !(_DWORD)v17 )
      goto LABEL_22;
    v11 = v17 - 1;
  }
  v12 = v11 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( v7 == *v13 )
    goto LABEL_4;
  v22 = 1;
  while ( v14 != -4096 )
  {
    v42 = v22 + 1;
    v12 = v11 & (v22 + v12);
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v7 == *v13 )
      goto LABEL_4;
    v22 = v42;
  }
  if ( v9 )
  {
    v21 = 64;
    goto LABEL_23;
  }
  v17 = *(unsigned int *)(a5 + 24);
LABEL_22:
  v21 = 16 * v17;
LABEL_23:
  v13 = (__int64 *)(v10 + v21);
LABEL_4:
  v15 = 64;
  if ( !v9 )
    v15 = 16LL * *(unsigned int *)(a5 + 24);
  if ( v13 != (__int64 *)(v10 + v15) )
    return v13[1];
  if ( *(_BYTE *)v7 <= 0x1Cu )
  {
    *sub_29260A0(a5, &v47) = v7;
    return v47;
  }
  else if ( a3 == *(_QWORD *)(v7 + 40) )
  {
    if ( *(_BYTE *)v7 == 84 )
    {
      v18 = *(_QWORD *)(v7 - 8);
      v19 = 0x1FFFFFFFE0LL;
      if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
      {
        v20 = 0;
        do
        {
          if ( a4 == *(_QWORD *)(v18 + 32LL * *(unsigned int *)(v7 + 72) + 8 * v20) )
          {
            v19 = 32 * v20;
            goto LABEL_19;
          }
          ++v20;
        }
        while ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != (_DWORD)v20 );
        v19 = 0x1FFFFFFFE0LL;
      }
LABEL_19:
      v7 = *(_QWORD *)(v18 + v19);
      *sub_29260A0(a5, &v47) = v7;
    }
    else
    {
      v23 = sub_B47F80((_BYTE *)v7);
      v48[0] = sub_BD5D20(v7);
      v48[2] = ".st.speculate";
      v49 = 773;
      v48[1] = v24;
      sub_BD6B50((unsigned __int8 *)v23, v48);
      v25 = 0;
      v26 = a4;
      v27 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
      if ( v27 )
      {
        do
        {
          if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
            v28 = *(_QWORD *)(v7 - 8);
          else
            v28 = v7 - 32LL * v27;
          v43 = v25;
          v46 = v26;
          v44 = 32LL * v25;
          v29 = sub_2926390(*(_QWORD *)(v28 + v44), a2, a3, v26, a5);
          v26 = v46;
          if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
            v30 = *(_QWORD *)(v23 - 8);
          else
            v30 = v23 - 32LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF);
          v31 = v30 + v44;
          if ( *(_QWORD *)(v30 + v44) )
          {
            v32 = *(_QWORD *)(v31 + 8);
            **(_QWORD **)(v31 + 16) = v32;
            if ( v32 )
              *(_QWORD *)(v32 + 16) = *(_QWORD *)(v31 + 16);
          }
          *(_QWORD *)v31 = v29;
          if ( v29 )
          {
            v33 = *(_QWORD *)(v29 + 16);
            *(_QWORD *)(v31 + 8) = v33;
            if ( v33 )
              *(_QWORD *)(v33 + 16) = v31 + 8;
            *(_QWORD *)(v31 + 16) = v29 + 16;
            *(_QWORD *)(v29 + 16) = v31;
          }
          v25 = v43 + 1;
          v27 = *(_DWORD *)(v7 + 4) & 0x7FFFFFF;
        }
        while ( v43 + 1 != v27 );
      }
      v34 = a2[11];
      v35 = a2[7];
      v36 = a2[8];
      v49 = 257;
      (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v34 + 16LL))(
        v34,
        v23,
        v48,
        v35,
        v36);
      v37 = 16LL * *((unsigned int *)a2 + 2);
      v38 = *a2;
      v39 = v38 + v37;
      while ( v39 != v38 )
      {
        v40 = *(_QWORD *)(v38 + 8);
        v41 = *(_DWORD *)v38;
        v38 += 16;
        sub_B99FD0(v23, v41, v40);
      }
      v7 = v23;
      *sub_29260A0(a5, &v47) = v23;
    }
  }
  else
  {
    *sub_29260A0(a5, &v47) = v7;
  }
  return v7;
}
