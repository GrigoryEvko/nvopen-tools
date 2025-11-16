// Function: sub_29B2CD0
// Address: 0x29b2cd0
//
__int64 __fastcall sub_29B2CD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 result; // rax
  __int64 v6; // r14
  __int64 *v8; // r13
  __int64 v9; // r12
  __int64 v10; // r8
  char *v11; // rbx
  char *v12; // r8
  __int64 *v13; // r12
  unsigned __int8 **v14; // r13
  __int64 v15; // rbx
  unsigned __int8 **v16; // r14
  int v17; // ecx
  unsigned int v18; // edx
  char *v19; // r9
  unsigned __int8 *v20; // rax
  int v21; // edx
  __int64 v22; // rdi
  unsigned __int8 v23; // dl
  __int64 v24; // rcx
  int v25; // eax
  __int64 v26; // rdi
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // r9
  int v30; // r10d
  __int64 v31; // rax
  __int64 v32; // rsi
  int v33; // edx
  __int64 v34; // rdi
  int v35; // ecx
  unsigned int v36; // edx
  __int64 v37; // r8
  __int64 v38; // rdx
  int v39; // r10d
  int v40; // r9d
  __int64 v41; // [rsp+0h] [rbp-80h]
  __int64 v43; // [rsp+10h] [rbp-70h]
  __int64 v45; // [rsp+20h] [rbp-60h]
  __int64 v46; // [rsp+28h] [rbp-58h]
  char *v48; // [rsp+38h] [rbp-48h]
  char *v49; // [rsp+48h] [rbp-38h] BYREF

  result = *(_QWORD *)(a1 + 88);
  v43 = result;
  v41 = result + 8LL * *(unsigned int *)(a1 + 96);
  if ( result != v41 )
  {
    v6 = a1;
    v8 = (__int64 *)&v49;
    while ( 1 )
    {
      v9 = *(_QWORD *)(*(_QWORD *)v43 + 56LL);
      v45 = *(_QWORD *)v43 + 48LL;
      if ( v9 != v45 )
        break;
LABEL_28:
      v43 += 8;
      result = v43;
      if ( v41 == v43 )
        return result;
    }
    while ( 1 )
    {
      if ( !v9 )
        BUG();
      v48 = (char *)(v9 - 24);
      v10 = 32LL * (*(_DWORD *)(v9 - 20) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v9 - 17) & 0x40) != 0 )
      {
        v11 = *(char **)(v9 - 32);
        v12 = &v11[v10];
      }
      else
      {
        v11 = &v48[-v10];
        v12 = (char *)(v9 - 24);
      }
      if ( v11 != v12 )
        break;
LABEL_20:
      v31 = *(_QWORD *)(v9 - 8);
      if ( v31 )
      {
        while ( 1 )
        {
          v38 = *(_QWORD *)(v31 + 24);
          if ( *(_BYTE *)v38 <= 0x1Cu )
            break;
          v32 = *(_QWORD *)(v38 + 40);
          v33 = *(_DWORD *)(v6 + 80);
          v34 = *(_QWORD *)(v6 + 64);
          if ( !v33 )
            break;
          v35 = v33 - 1;
          v36 = (v33 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
          v37 = *(_QWORD *)(v34 + 8LL * v36);
          if ( v37 != v32 )
          {
            v40 = 1;
            while ( v37 != -4096 )
            {
              v36 = v35 & (v40 + v36);
              v37 = *(_QWORD *)(v34 + 8LL * v36);
              if ( v32 == v37 )
                goto LABEL_24;
              ++v40;
            }
            break;
          }
LABEL_24:
          v31 = *(_QWORD *)(v31 + 8);
          if ( !v31 )
            goto LABEL_27;
        }
        v49 = v48;
        sub_29B2110(a3, v8);
      }
LABEL_27:
      v9 = *(_QWORD *)(v9 + 8);
      if ( v45 == v9 )
        goto LABEL_28;
    }
    v46 = v9;
    v13 = v8;
    v14 = (unsigned __int8 **)v11;
    v15 = v6;
    v16 = (unsigned __int8 **)v12;
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = *v14;
        v21 = *(_DWORD *)(a4 + 24);
        v22 = *(_QWORD *)(a4 + 8);
        v49 = (char *)*v14;
        if ( v21 )
        {
          v17 = v21 - 1;
          v18 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v19 = *(char **)(v22 + 8LL * v18);
          if ( v20 == (unsigned __int8 *)v19 )
            goto LABEL_10;
          v39 = 1;
          while ( v19 != (char *)-4096LL )
          {
            v18 = v17 & (v39 + v18);
            v19 = *(char **)(v22 + 8LL * v18);
            if ( v20 == (unsigned __int8 *)v19 )
              goto LABEL_10;
            ++v39;
          }
        }
        v23 = *v20;
        if ( *v20 != 22 )
          break;
LABEL_18:
        v14 += 4;
        sub_29B2110(a2, v13);
        if ( v16 == v14 )
        {
LABEL_19:
          v8 = v13;
          v9 = v46;
          v6 = v15;
          goto LABEL_20;
        }
      }
      if ( v23 <= 0x1Cu )
      {
        if ( v23 != 3 || !a5 )
          goto LABEL_10;
        goto LABEL_18;
      }
      v24 = *((_QWORD *)v20 + 5);
      v25 = *(_DWORD *)(v15 + 80);
      v26 = *(_QWORD *)(v15 + 64);
      if ( !v25 )
        goto LABEL_18;
      v27 = v25 - 1;
      v28 = (v25 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v29 = *(_QWORD *)(v26 + 8LL * v28);
      if ( v24 != v29 )
      {
        v30 = 1;
        while ( v29 != -4096 )
        {
          v28 = v27 & (v30 + v28);
          v29 = *(_QWORD *)(v26 + 8LL * v28);
          if ( v24 == v29 )
            goto LABEL_10;
          ++v30;
        }
        goto LABEL_18;
      }
LABEL_10:
      v14 += 4;
      if ( v16 == v14 )
        goto LABEL_19;
    }
  }
  return result;
}
