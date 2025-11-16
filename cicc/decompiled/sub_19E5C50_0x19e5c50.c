// Function: sub_19E5C50
// Address: 0x19e5c50
//
void __fastcall sub_19E5C50(__int64 a1, __int64 a2)
{
  __int64 i; // r14
  __int64 v5; // rdi
  int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rsi
  int v15; // ecx
  int v16; // ecx
  unsigned int v17; // edi
  __int64 v18; // rdx
  _QWORD *v19; // r8
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // rcx
  int v23; // r8d
  unsigned int v24; // edx
  __int64 *v25; // r12
  __int64 v26; // rsi
  __int64 v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  _QWORD *v31; // rcx
  int v32; // eax
  __int64 v33; // rcx
  int v34; // edi
  __int64 v35; // rsi
  unsigned int v36; // edx
  __int64 *v37; // rax
  __int64 v38; // r8
  unsigned int v39; // ecx
  __int64 v40; // rdx
  __int64 v41; // rax
  _QWORD *v42; // rax
  unsigned __int64 v43; // rdi
  int v44; // eax
  int v45; // edx
  int v46; // r9d
  int v47; // r9d
  int v48; // r9d
  __int64 v49; // [rsp-78h] [rbp-78h] BYREF
  __int64 *v50; // [rsp-70h] [rbp-70h] BYREF
  _QWORD *v51; // [rsp-68h] [rbp-68h] BYREF
  _QWORD *v52; // [rsp-60h] [rbp-60h]
  __int64 *v53; // [rsp-58h] [rbp-58h]
  __int64 v54; // [rsp-50h] [rbp-50h]
  _QWORD v55[9]; // [rsp-48h] [rbp-48h] BYREF

  if ( *(_BYTE *)(a2 + 16) == 21 )
    return;
  for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
  {
    v13 = sub_1648700(i);
    v14 = *(_QWORD *)(a1 + 2400);
    v15 = *(_DWORD *)(a1 + 2416);
    if ( (unsigned int)*((unsigned __int8 *)v13 + 16) - 21 <= 1 )
    {
      if ( !v15 )
        goto LABEL_15;
      v5 = v13[9];
      v6 = v15 - 1;
      v7 = v6 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v8 = (__int64 *)(v14 + 16LL * v7);
      v9 = *v8;
      if ( v5 != *v8 )
      {
        v20 = 1;
        while ( v9 != -8 )
        {
          v47 = v20 + 1;
          v7 = v6 & (v20 + v7);
          v8 = (__int64 *)(v14 + 16LL * v7);
          v9 = *v8;
          if ( v5 == *v8 )
            goto LABEL_6;
          v20 = v47;
        }
        goto LABEL_15;
      }
LABEL_6:
      v10 = *((_DWORD *)v8 + 2);
    }
    else
    {
      if ( !v15 )
        goto LABEL_15;
      v16 = v15 - 1;
      v17 = v16 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v18 = v14 + 16LL * v17;
      v19 = *(_QWORD **)v18;
      if ( v13 != *(_QWORD **)v18 )
      {
        v45 = 1;
        while ( v19 != (_QWORD *)-8LL )
        {
          v46 = v45 + 1;
          v17 = v16 & (v45 + v17);
          v18 = v14 + 16LL * v17;
          v19 = *(_QWORD **)v18;
          if ( v13 == *(_QWORD **)v18 )
            goto LABEL_12;
          v45 = v46;
        }
LABEL_15:
        v11 = 1;
        v12 = 0;
        goto LABEL_8;
      }
LABEL_12:
      v10 = *(_DWORD *)(v18 + 8);
    }
    v11 = 1LL << v10;
    v12 = 8LL * (v10 >> 6);
LABEL_8:
    *(_QWORD *)(*(_QWORD *)(a1 + 2336) + v12) |= v11;
  }
  v21 = *(unsigned int *)(a1 + 1952);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD *)(a1 + 1936);
    v23 = 1;
    v24 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v25 = (__int64 *)(v22 + ((unsigned __int64)v24 << 6));
    v26 = *v25;
    if ( *v25 == a2 )
    {
LABEL_18:
      if ( v25 != (__int64 *)(v22 + (v21 << 6)) )
      {
        v27 = v25[3];
        if ( v27 == v25[2] )
          v28 = (_QWORD *)(v27 + 8LL * *((unsigned int *)v25 + 9));
        else
          v28 = (_QWORD *)(v27 + 8LL * *((unsigned int *)v25 + 8));
        v51 = (_QWORD *)v25[3];
        v52 = v28;
        sub_19E4730((__int64)&v51);
        v53 = v25 + 1;
        v54 = v25[1];
        v29 = v25[3];
        if ( v29 == v25[2] )
          v30 = *((unsigned int *)v25 + 9);
        else
          v30 = *((unsigned int *)v25 + 8);
        v55[0] = v29 + 8 * v30;
        v55[1] = v55[0];
        sub_19E4730((__int64)v55);
        v55[2] = v25 + 1;
        v31 = v51;
        v55[3] = v25[1];
        if ( (_QWORD *)v55[0] != v51 )
        {
          while ( 1 )
          {
            if ( (unsigned int)*(unsigned __int8 *)(*v31 + 16LL) - 21 > 1 )
            {
              v49 = *v31;
              if ( !(unsigned __int8)sub_154CC80(a1 + 2392, &v49, &v50) )
                goto LABEL_38;
              v37 = v50;
            }
            else
            {
              v32 = *(_DWORD *)(a1 + 2416);
              if ( !v32 )
                goto LABEL_38;
              v33 = *(_QWORD *)(*v31 + 72LL);
              v34 = v32 - 1;
              v35 = *(_QWORD *)(a1 + 2400);
              v36 = (v32 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
              v37 = (__int64 *)(v35 + 16LL * v36);
              v38 = *v37;
              if ( *v37 != v33 )
              {
                v44 = 1;
                while ( v38 != -8 )
                {
                  v48 = v44 + 1;
                  v36 = v34 & (v44 + v36);
                  v37 = (__int64 *)(v35 + 16LL * v36);
                  v38 = *v37;
                  if ( v33 == *v37 )
                    goto LABEL_27;
                  v44 = v48;
                }
LABEL_38:
                v40 = 1;
                v41 = 0;
                goto LABEL_28;
              }
            }
LABEL_27:
            v39 = *((_DWORD *)v37 + 2);
            v40 = 1LL << v39;
            v41 = 8LL * (v39 >> 6);
LABEL_28:
            *(_QWORD *)(*(_QWORD *)(a1 + 2336) + v41) |= v40;
            v31 = v52;
            v42 = v51 + 1;
            v51 = v42;
            if ( v42 == v52 )
            {
LABEL_31:
              if ( (_QWORD *)v55[0] == v52 )
                break;
            }
            else
            {
              while ( (unsigned __int64)(*v42 + 2LL) <= 1 )
              {
                v51 = ++v42;
                if ( v42 == v52 )
                  goto LABEL_31;
              }
              v31 = v51;
              if ( (_QWORD *)v55[0] == v51 )
                break;
            }
          }
        }
        v43 = v25[3];
        if ( v43 != v25[2] )
          _libc_free(v43);
        *v25 = -16;
        --*(_DWORD *)(a1 + 1944);
        ++*(_DWORD *)(a1 + 1948);
      }
    }
    else
    {
      while ( v26 != -8 )
      {
        v24 = (v21 - 1) & (v23 + v24);
        v25 = (__int64 *)(v22 + ((unsigned __int64)v24 << 6));
        v26 = *v25;
        if ( *v25 == a2 )
          goto LABEL_18;
        ++v23;
      }
    }
  }
}
