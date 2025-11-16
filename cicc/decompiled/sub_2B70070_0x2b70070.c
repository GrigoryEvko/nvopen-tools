// Function: sub_2B70070
// Address: 0x2b70070
//
unsigned __int64 __fastcall sub_2B70070(int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  char v13; // si
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // r10d
  __int64 v21; // rdx
  int v22; // ecx
  unsigned __int8 **v23; // rsi
  unsigned __int8 **v24; // rax
  unsigned __int8 *v25; // rcx
  __int64 v26; // rax
  bool v27; // of
  __int64 v28; // r13
  int v29; // edi
  int *v30; // rdx
  int v31; // edi
  int v32; // ecx
  unsigned __int8 **v33; // rdx
  __int64 v34; // r8
  int v35; // r8d
  int v36; // r11d
  __int64 v37; // [rsp+0h] [rbp-60h]
  int v38; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 v39; // [rsp+18h] [rbp-48h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  int v41; // [rsp+28h] [rbp-38h]
  char v42; // [rsp+2Ch] [rbp-34h] BYREF

  v6 = a2;
  if ( *a1 != a1[1] && *a1 == 49 )
    return v6;
  v10 = *((_QWORD *)a1 + 1);
  if ( (*(_BYTE *)(v10 + 7) & 0x40) != 0 )
  {
    v11 = *((_QWORD *)a1 + 2);
    v12 = **(_QWORD **)(v10 - 8);
    v13 = *(_BYTE *)(v11 + 88) & 1;
    if ( v13 )
      goto LABEL_6;
  }
  else
  {
    v11 = *((_QWORD *)a1 + 2);
    v12 = *(_QWORD *)(v10 - 32LL * (*(_DWORD *)(v10 + 4) & 0x7FFFFFF));
    v13 = *(_BYTE *)(v11 + 88) & 1;
    if ( v13 )
    {
LABEL_6:
      v14 = v11 + 96;
      a6 = 3;
      goto LABEL_7;
    }
  }
  v15 = *(unsigned int *)(v11 + 104);
  v14 = *(_QWORD *)(v11 + 96);
  if ( !(_DWORD)v15 )
    goto LABEL_46;
  a6 = (unsigned int)(v15 - 1);
LABEL_7:
  v15 = (unsigned int)a6 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v16 = v14 + 72 * v15;
  v17 = *(_QWORD *)v16;
  if ( v12 == *(_QWORD *)v16 )
    goto LABEL_8;
  v35 = 1;
  while ( v17 != -4096 )
  {
    v36 = v35 + 1;
    v15 = (unsigned int)a6 & ((_DWORD)v15 + v35);
    v16 = v14 + 72LL * (unsigned int)v15;
    v17 = *(_QWORD *)v16;
    if ( v12 == *(_QWORD *)v16 )
      goto LABEL_8;
    v35 = v36;
  }
  if ( v13 )
  {
    v34 = 288;
    goto LABEL_47;
  }
  v15 = *(unsigned int *)(v11 + 104);
LABEL_46:
  v34 = 72 * v15;
LABEL_47:
  v16 = v14 + v34;
LABEL_8:
  v18 = 288;
  if ( !v13 )
    v18 = 72LL * *(unsigned int *)(v11 + 104);
  if ( v16 != v14 + v18 && *(_DWORD *)(v16 + 16) == 1 )
  {
    sub_2B2D870(v11, **(_QWORD **)(v16 + 8), v15, v14, v16, a6);
  }
  else
  {
    v19 = *(_QWORD *)(**((_QWORD **)a1 + 3) + 240LL);
    sub_2B5F980(*(__int64 **)v19, *(unsigned int *)(v19 + 8), *(__int64 **)(v11 + 3304));
  }
  v20 = *a1;
  v37 = *((_QWORD *)a1 + 5);
  if ( !*(_DWORD *)(*((_QWORD *)a1 + 4) + 200LL) )
  {
    v21 = *(_QWORD *)(v37 + 3272);
    if ( v21 )
    {
      v22 = *(_DWORD *)(v21 + 8) >> 1;
      if ( (*(_BYTE *)(v21 + 8) & 1) != 0 )
      {
        v23 = (unsigned __int8 **)(v21 + 48);
        v24 = (unsigned __int8 **)(v21 + 16);
        if ( !v22 )
          goto LABEL_21;
      }
      else
      {
        v24 = *(unsigned __int8 ***)(v21 + 16);
        v23 = &v24[*(unsigned int *)(v21 + 24)];
        if ( !v22 )
          goto LABEL_21;
      }
      if ( v24 != v23 )
      {
        while ( 1 )
        {
          v25 = *v24;
          if ( *v24 != (unsigned __int8 *)-4096LL && v25 != (unsigned __int8 *)-8192LL )
            break;
          if ( ++v24 == v23 )
            goto LABEL_21;
        }
        if ( v24 != v23 )
        {
LABEL_29:
          v29 = *v25;
          v30 = &v38;
          v38 = 14;
          v39 = 0x1200000011LL;
          v40 = 0x1D0000001CLL;
          v31 = v29 - 29;
          v32 = 13;
          v41 = 30;
          while ( 1 )
          {
            if ( v31 == v32 )
            {
              v33 = v24 + 1;
              if ( v24 + 1 != v23 )
              {
                while ( 1 )
                {
                  v25 = *v33;
                  v24 = v33;
                  if ( *v33 != (unsigned __int8 *)-4096LL && v25 != (unsigned __int8 *)-8192LL )
                    break;
                  if ( v23 == ++v33 )
                  {
                    if ( v20 - 39 > 1 )
                      goto LABEL_22;
                    return v6;
                  }
                }
                if ( v33 != v23 )
                  goto LABEL_29;
              }
              goto LABEL_21;
            }
            if ( &v42 == (char *)v30 )
              break;
            v32 = *v30++;
          }
          if ( v24 != v23 )
            goto LABEL_22;
        }
      }
LABEL_21:
      if ( v20 - 39 <= 1 )
        return v6;
    }
  }
LABEL_22:
  v26 = sub_DFD060(*(__int64 **)(v37 + 3296), v20, *((_QWORD *)a1 + 6), *((_QWORD *)a1 + 7));
  v27 = __OFADD__(v26, a2);
  v28 = v26 + a2;
  if ( !v27 )
    return v28;
  v6 = 0x7FFFFFFFFFFFFFFFLL;
  if ( v26 <= 0 )
    return 0x8000000000000000LL;
  return v6;
}
