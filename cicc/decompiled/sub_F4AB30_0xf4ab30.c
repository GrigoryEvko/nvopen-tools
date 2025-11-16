// Function: sub_F4AB30
// Address: 0xf4ab30
//
__int64 __fastcall sub_F4AB30(__int64 a1, char *a2, _BYTE *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 i; // rbx
  __int64 v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  _QWORD *v13; // rdi
  __int64 v14; // rax
  const char *v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // edx
  unsigned __int64 v19; // rax
  bool v20; // cf
  unsigned __int64 v21; // rdx
  _BYTE *v22; // r14
  unsigned __int64 v23; // rax
  int v24; // edx
  _BYTE *v25; // rax
  __int64 v26; // r12
  const char *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rdx
  _QWORD *v30; // r14
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r10
  __int64 v34; // rax
  __int64 v35; // rax
  _BYTE *v36; // rdi
  __int64 v37; // r9
  unsigned int v38; // r8d
  __int64 v39; // rsi
  _BYTE *v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rsi
  int v43; // esi
  int v44; // r14d
  __int64 v46; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v47; // [rsp+8h] [rbp-98h]
  __int64 j; // [rsp+38h] [rbp-68h]
  __int64 v52; // [rsp+40h] [rbp-60h] BYREF
  __int64 v53; // [rsp+48h] [rbp-58h]
  char *v54; // [rsp+50h] [rbp-50h]
  unsigned __int64 v55; // [rsp+58h] [rbp-48h]
  __int64 v56; // [rsp+60h] [rbp-40h]
  unsigned __int64 v57; // [rsp+68h] [rbp-38h]

  for ( i = *(_QWORD *)(a1 + 56); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) != 84 )
      break;
    v9 = *(_QWORD *)(i - 32);
    v10 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
    {
      v11 = 0;
      do
      {
        if ( a2 == *(char **)(v9 + 32LL * *(unsigned int *)(i + 48) + 8 * v11) )
        {
          v10 = 32 * v11;
          goto LABEL_9;
        }
        ++v11;
      }
      while ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != (_DWORD)v11 );
      v10 = 0x1FFFFFFFE0LL;
    }
LABEL_9:
    v12 = *(_QWORD *)(v9 + v10);
    v13 = sub_F46C80(a4, i - 24);
    v14 = v13[2];
    if ( v14 != v12 )
    {
      if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
        sub_BD60C0(v13);
      v13[2] = v12;
      if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
        sub_BD73F0((__int64)v13);
    }
  }
  LOWORD(v56) = 257;
  v46 = sub_F41C30((__int64)a2, a1, 0, 0, 0, (void **)&v52);
  v15 = sub_BD5D20((__int64)a2);
  LOWORD(v56) = 773;
  v52 = (__int64)v15;
  v53 = v16;
  v54 = ".split";
  sub_BD6B50((unsigned __int8 *)v46, (const char **)&v52);
  v17 = *(_QWORD *)(v46 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v17 == v46 + 48 )
  {
    v47 = 0;
  }
  else
  {
    if ( !v17 )
      BUG();
    v18 = *(unsigned __int8 *)(v17 - 24);
    v19 = v17 - 24;
    v20 = (unsigned int)(v18 - 30) < 0xB;
    v21 = 0;
    if ( v20 )
      v21 = v19;
    v47 = v21;
  }
  v52 = (__int64)a2;
  v54 = a2;
  v56 = v46;
  v53 = a1 | 4;
  v57 = a1 & 0xFFFFFFFFFFFFFFFBLL;
  v55 = v46 & 0xFFFFFFFFFFFFFFFBLL;
  sub_FFB3D0(a5, &v52, 3);
  for ( j = i; ; j = *(_QWORD *)(j + 8) )
  {
    v22 = (_BYTE *)(j - 24);
    if ( !j )
      v22 = 0;
    if ( a3 == v22 )
      break;
    v23 = *(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v23 == a1 + 48 )
    {
      v25 = 0;
    }
    else
    {
      if ( !v23 )
        BUG();
      v24 = *(unsigned __int8 *)(v23 - 24);
      v25 = (_BYTE *)(v23 - 24);
      if ( (unsigned int)(v24 - 30) >= 0xB )
        v25 = 0;
    }
    if ( v22 == v25 )
      break;
    LOWORD(v5) = 0;
    v26 = sub_B47F80(v22);
    v27 = sub_BD5D20((__int64)v22);
    LOWORD(v56) = 261;
    v52 = (__int64)v27;
    v53 = v28;
    sub_BD6B50((unsigned __int8 *)v26, (const char **)&v52);
    sub_B44220((_QWORD *)v26, v47 + 24, v5);
    LOBYTE(v53) = 0;
    sub_B43F50(v26, (__int64)v22, v52, 0, 0);
    v30 = sub_F46C80(a4, (__int64)v22);
    v31 = v30[2];
    if ( v26 != v31 )
    {
      if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
        sub_BD60C0(v30);
      v30[2] = v26;
      LOBYTE(v29) = v26 != 0;
      if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
        sub_BD73F0((__int64)v30);
    }
    v32 = 0;
    v33 = 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
    if ( (*(_DWORD *)(v26 + 4) & 0x7FFFFFF) != 0 )
    {
      do
      {
        if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
        {
          v34 = *(_QWORD *)(v26 - 8);
        }
        else
        {
          v29 = 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
          v34 = v26 - v29;
        }
        v35 = v32 + v34;
        v36 = *(_BYTE **)v35;
        if ( **(_BYTE **)v35 > 0x1Cu )
        {
          v29 = *(unsigned int *)(a4 + 24);
          if ( (_DWORD)v29 )
          {
            v37 = *(_QWORD *)(a4 + 8);
            v38 = (v29 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
            v39 = v37 + ((unsigned __int64)v38 << 6);
            v40 = *(_BYTE **)(v39 + 24);
            if ( v36 == v40 )
            {
LABEL_44:
              v29 = v37 + (v29 << 6);
              if ( v39 != v29 )
              {
                v29 = *(_QWORD *)(v39 + 56);
                v41 = *(_QWORD *)(v35 + 8);
                **(_QWORD **)(v35 + 16) = v41;
                if ( v41 )
                  *(_QWORD *)(v41 + 16) = *(_QWORD *)(v35 + 16);
                *(_QWORD *)v35 = v29;
                if ( v29 )
                {
                  v42 = *(_QWORD *)(v29 + 16);
                  *(_QWORD *)(v35 + 8) = v42;
                  if ( v42 )
                    *(_QWORD *)(v42 + 16) = v35 + 8;
                  *(_QWORD *)(v35 + 16) = v29 + 16;
                  *(_QWORD *)(v29 + 16) = v35;
                }
              }
            }
            else
            {
              v43 = 1;
              while ( v40 != (_BYTE *)-4096LL )
              {
                v44 = v43 + 1;
                v38 = (v29 - 1) & (v43 + v38);
                v39 = v37 + ((unsigned __int64)v38 << 6);
                v40 = *(_BYTE **)(v39 + 24);
                if ( v36 == v40 )
                  goto LABEL_44;
                v43 = v44;
              }
            }
          }
        }
        v32 += 32;
      }
      while ( v32 != v33 );
    }
    sub_F581B0(a4, v26, v29, v32);
  }
  return v46;
}
