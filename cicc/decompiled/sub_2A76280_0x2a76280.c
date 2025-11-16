// Function: sub_2A76280
// Address: 0x2a76280
//
__int64 __fastcall sub_2A76280(__int64 a1, __int64 a2)
{
  unsigned int v4; // r13d
  __int64 *v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // edx
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // rax
  __int64 v19; // r14
  __int64 v20; // r8
  __int64 v21; // rsi
  __int64 v22; // rax
  int v23; // ecx
  __int64 v24; // rdi
  int v25; // ecx
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  _QWORD *v29; // rdx
  unsigned int v30; // r8d
  __int64 *v31; // rax
  __int64 v32; // r9
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rbx
  __int64 v36; // rdx
  int v37; // eax
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rsi
  unsigned __int64 *v42; // r14
  unsigned __int64 *v43; // rdi
  int v44; // r12d
  int v45; // eax
  int v46; // r10d
  int v47; // eax
  int v48; // r10d
  __int64 *v49; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v50[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v51[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( !sub_D97040(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 + 8)) )
    return 0;
  v6 = sub_DD8400(*(_QWORD *)(a1 + 16), a2);
  v7 = *(_QWORD *)(a1 + 16);
  v8 = *(_QWORD *)a1;
  v49 = v6;
  if ( !sub_DADE90(v7, (__int64)v6, v8)
    || (unsigned __int8)sub_F6CE90(
                          *(_QWORD *)(a1 + 40),
                          (__int64 *)&v49,
                          1,
                          *(_QWORD *)a1,
                          qword_4F8C268[8],
                          *(_QWORD *)(a1 + 32),
                          a2) )
  {
    return 0;
  }
  v9 = sub_D4B130(*(_QWORD *)a1);
  if ( v9 )
  {
    v13 = v9 + 48;
    v14 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v14 == v13 )
    {
      v16 = 0;
    }
    else
    {
      if ( !v14 )
        BUG();
      v15 = *(unsigned __int8 *)(v14 - 24);
      v16 = 0;
      v17 = v14 - 24;
      if ( (unsigned int)(v15 - 30) < 0xB )
        v16 = v17;
    }
  }
  else
  {
    v16 = a2;
  }
  v4 = sub_F80650(*(__int64 **)(a1 + 40), (__int64)v49, v16, v10, v11, v12);
  if ( !(_BYTE)v4 )
    return 0;
  v18 = sub_F8DB90(*(_QWORD *)(a1 + 40), (__int64)v49, *(_QWORD *)(a2 + 8), v16 + 24, 0);
  v19 = (__int64)v18;
  if ( *(_BYTE *)v18 > 0x1Cu )
  {
    v20 = v18[5];
    v21 = *(_QWORD *)(a2 + 40);
    if ( v20 != v21 )
    {
      v22 = *(_QWORD *)(a1 + 8);
      v23 = *(_DWORD *)(v22 + 24);
      v24 = *(_QWORD *)(v22 + 8);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = v25 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v27 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v27;
        if ( v20 == *v27 )
        {
LABEL_16:
          v29 = (_QWORD *)v27[1];
          if ( v29 )
          {
            v30 = v25 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v31 = (__int64 *)(v24 + 16LL * v30);
            v32 = *v31;
            if ( v21 != *v31 )
            {
              v47 = 1;
              while ( v32 != -4096 )
              {
                v48 = v47 + 1;
                v30 = v25 & (v47 + v30);
                v31 = (__int64 *)(v24 + 16LL * v30);
                v32 = *v31;
                if ( v21 == *v31 )
                  goto LABEL_18;
                v47 = v48;
              }
              goto LABEL_30;
            }
LABEL_18:
            v33 = (_QWORD *)v31[1];
            if ( v33 != v29 )
            {
              while ( v33 )
              {
                v33 = (_QWORD *)*v33;
                if ( v33 == v29 )
                  goto LABEL_21;
              }
LABEL_30:
              sub_BD84D0(a2, v19);
              v39 = *(_QWORD *)(a1 + 16);
              v40 = *(_QWORD *)(a1 + 8);
              v41 = *(_QWORD *)(a1 + 24);
              v50[0] = (unsigned __int64)v51;
              v51[0] = v19;
              v50[1] = 0x100000001LL;
              sub_11D0BA0((__int64)v50, v41, v40, v39, 0, 0);
              if ( (_QWORD *)v50[0] != v51 )
                _libc_free(v50[0]);
              goto LABEL_22;
            }
          }
        }
        else
        {
          v45 = 1;
          while ( v28 != -4096 )
          {
            v46 = v45 + 1;
            v26 = v25 & (v45 + v26);
            v27 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v27;
            if ( v20 == *v27 )
              goto LABEL_16;
            v45 = v46;
          }
        }
      }
    }
  }
LABEL_21:
  sub_BD84D0(a2, v19);
LABEL_22:
  *(_BYTE *)(a1 + 56) = 1;
  v35 = *(_QWORD *)(a1 + 48);
  v36 = *(unsigned int *)(v35 + 8);
  v37 = v36;
  if ( *(_DWORD *)(v35 + 12) <= (unsigned int)v36 )
  {
    v42 = (unsigned __int64 *)sub_C8D7D0(v35, v35 + 16, 0, 0x18u, v50, v34);
    v43 = &v42[3 * *(unsigned int *)(v35 + 8)];
    if ( v43 )
    {
      *v43 = 6;
      v43[1] = 0;
      v43[2] = a2;
      if ( a2 != -8192 && a2 != -4096 )
        sub_BD73F0((__int64)v43);
    }
    sub_F17F80(v35, v42);
    v44 = v50[0];
    if ( v35 + 16 != *(_QWORD *)v35 )
      _libc_free(*(_QWORD *)v35);
    ++*(_DWORD *)(v35 + 8);
    *(_QWORD *)v35 = v42;
    *(_DWORD *)(v35 + 12) = v44;
  }
  else
  {
    v38 = (_QWORD *)(*(_QWORD *)v35 + 24 * v36);
    if ( v38 )
    {
      *v38 = 6;
      v38[1] = 0;
      v38[2] = a2;
      if ( a2 != -4096 && a2 != -8192 )
        sub_BD73F0((__int64)v38);
      v37 = *(_DWORD *)(v35 + 8);
    }
    *(_DWORD *)(v35 + 8) = v37 + 1;
  }
  return v4;
}
