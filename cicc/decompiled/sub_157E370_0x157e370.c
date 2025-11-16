// Function: sub_157E370
// Address: 0x157e370
//
__int64 __fastcall sub_157E370(__int64 *a1)
{
  __int64 *v1; // r13
  unsigned int v2; // eax
  unsigned int v3; // ebx
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 *v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r14
  unsigned __int64 *v10; // rax
  unsigned int v11; // r12d
  unsigned int v12; // eax
  __int64 *v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // eax
  __int64 *v18; // r9
  unsigned int v19; // ecx
  unsigned int v20; // edi
  unsigned int v21; // edx
  __int64 v22; // rax
  char v23; // bl
  _QWORD *v24; // rax
  _QWORD *v25; // rcx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  int v31; // r10d
  unsigned int v32; // edx
  __int64 v33; // rdi
  int v34; // r8d
  __int64 *v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rdi
  int v38; // r8d
  unsigned __int8 v39; // [rsp+18h] [rbp-E8h]
  char v40; // [rsp+2Fh] [rbp-D1h] BYREF
  const char *v41; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v42; // [rsp+38h] [rbp-C8h]
  __int64 *v43; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v44; // [rsp+48h] [rbp-B8h]
  __int64 *v45; // [rsp+80h] [rbp-80h] BYREF
  __int64 v46; // [rsp+88h] [rbp-78h]
  _BYTE v47[112]; // [rsp+90h] [rbp-70h] BYREF

  v1 = a1;
  v2 = sub_15AC060();
  if ( v2 == 3 )
  {
    v41 = "nvvmir.version";
    v6 = (__int64)&v41;
    LOWORD(v43) = 259;
    v9 = sub_1632310(a1, &v41);
    if ( !v9 )
      goto LABEL_49;
    v41 = 0;
    v10 = (unsigned __int64 *)&v43;
    v42 = 1;
    do
      *v10++ = -8;
    while ( v10 != (unsigned __int64 *)&v45 );
    v11 = 0;
    v45 = (__int64 *)v47;
    v46 = 0x800000000LL;
    while ( 1 )
    {
      if ( (unsigned int)sub_161F520(v9, v6, v7, v8) <= v11 )
      {
        a1 = &v45[(unsigned int)v46];
        if ( v45 != a1 )
        {
          v7 = v45;
          v23 = 0;
          while ( 1 )
          {
            v26 = *v7;
            if ( !*v7 )
              break;
            if ( *(_DWORD *)(v26 + 8) == 4 )
            {
              v6 = *(_QWORD *)(v26 - 16);
              v27 = 0;
              if ( *(_BYTE *)v6 == 1 )
              {
                v27 = *(_QWORD *)(v6 + 136);
                if ( *(_BYTE *)(v27 + 16) != 13 )
                  v27 = 0;
              }
              v28 = *(_QWORD *)(v26 - 8);
              if ( *(_BYTE *)v28 != 1 )
                break;
              v6 = *(_QWORD *)(v28 + 136);
              if ( *(_BYTE *)(v6 + 16) != 13 || !v27 )
                break;
              v24 = *(_QWORD **)(v27 + 24);
              if ( *(_DWORD *)(v27 + 32) > 0x40u )
                v24 = (_QWORD *)*v24;
              v25 = *(_QWORD **)(v6 + 24);
              if ( *(_DWORD *)(v6 + 32) > 0x40u )
                v25 = (_QWORD *)*v25;
              if ( v24 != (_QWORD *)3 || (unsigned __int64)v25 > 2 )
                break;
              v23 = 1;
            }
            if ( a1 == ++v7 )
              goto LABEL_43;
          }
        }
        v23 = 0;
LABEL_43:
        if ( v45 != (__int64 *)v47 )
        {
          a1 = v45;
          _libc_free((unsigned __int64)v45);
        }
        if ( (v42 & 1) != 0 )
        {
          if ( v23 )
            return 0;
LABEL_49:
          v40 = 0;
          v29 = sub_16E8CB0(a1, v6, v7);
          if ( (unsigned __int8)sub_166CBC0(v1, v29, &v40) )
            sub_16BD130("Broken module found, compilation aborted!", 1);
          if ( v40 )
          {
            v30 = *v1;
            v43 = v1;
            v42 = 0x100000004LL;
            v41 = (const char *)&unk_49ECEF0;
            sub_16027F0(v30, &v41);
            return sub_15ACB40(v1);
          }
        }
        else
        {
          a1 = v43;
          j___libc_free_0(v43);
          if ( !v23 )
            goto LABEL_49;
        }
        return 0;
      }
      v15 = sub_161F530(v9, v11);
      v16 = v15;
      v7 = (__int64 *)(v42 & 1);
      if ( (v42 & 1) != 0 )
      {
        v6 = 7;
        v8 = (__int64)&v43;
      }
      else
      {
        v6 = v44;
        v8 = (__int64)v43;
        if ( !v44 )
        {
          v17 = v42;
          ++v41;
          v18 = 0;
          v19 = ((unsigned int)v42 >> 1) + 1;
LABEL_16:
          v20 = 3 * v6;
          goto LABEL_17;
        }
        v6 = v44 - 1;
      }
      v12 = v6 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v13 = (__int64 *)(v8 + 8LL * ((unsigned int)v6 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4))));
      v14 = *v13;
      if ( v16 == *v13 )
        goto LABEL_10;
      v31 = 1;
      v18 = 0;
      while ( v14 != -8 )
      {
        if ( v18 || v14 != -16 )
          v13 = v18;
        v12 = v6 & (v31 + v12);
        v14 = *(_QWORD *)(v8 + 8LL * v12);
        if ( v16 == v14 )
          goto LABEL_10;
        ++v31;
        v18 = v13;
        v13 = (__int64 *)(v8 + 8LL * v12);
      }
      v17 = v42;
      if ( !v18 )
        v18 = v13;
      ++v41;
      v19 = ((unsigned int)v42 >> 1) + 1;
      if ( !(_BYTE)v7 )
      {
        v6 = v44;
        goto LABEL_16;
      }
      v20 = 24;
      v6 = 8;
LABEL_17:
      if ( v20 <= 4 * v19 )
      {
        sub_12BFBB0((__int64)&v41, 2 * v6);
        if ( (v42 & 1) != 0 )
        {
          v8 = 7;
          v6 = (__int64)&v43;
        }
        else
        {
          v6 = (__int64)v43;
          if ( !v44 )
            goto LABEL_92;
          v8 = v44 - 1;
        }
        v32 = v8 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v18 = (__int64 *)(v6 + 8LL * v32);
        v17 = v42;
        v33 = *v18;
        if ( v16 == *v18 )
          goto LABEL_19;
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( !v35 && v33 == -16 )
            v35 = v18;
          v32 = v8 & (v34 + v32);
          v18 = (__int64 *)(v6 + 8LL * v32);
          v33 = *v18;
          if ( v16 == *v18 )
            goto LABEL_66;
          ++v34;
        }
      }
      else
      {
        v21 = v6 - HIDWORD(v42) - v19;
        v8 = (unsigned int)v6 >> 3;
        if ( v21 > (unsigned int)v8 )
          goto LABEL_19;
        sub_12BFBB0((__int64)&v41, v6);
        if ( (v42 & 1) != 0 )
        {
          v8 = 7;
          v6 = (__int64)&v43;
        }
        else
        {
          v6 = (__int64)v43;
          if ( !v44 )
          {
LABEL_92:
            LODWORD(v42) = (2 * ((unsigned int)v42 >> 1) + 2) | v42 & 1;
            BUG();
          }
          v8 = v44 - 1;
        }
        v36 = v8 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v18 = (__int64 *)(v6 + 8LL * v36);
        v17 = v42;
        v37 = *v18;
        if ( v16 == *v18 )
          goto LABEL_19;
        v38 = 1;
        v35 = 0;
        while ( v37 != -8 )
        {
          if ( v37 == -16 && !v35 )
            v35 = v18;
          v36 = v8 & (v38 + v36);
          v18 = (__int64 *)(v6 + 8LL * v36);
          v37 = *v18;
          if ( v16 == *v18 )
            goto LABEL_66;
          ++v38;
        }
      }
      if ( v35 )
        v18 = v35;
LABEL_66:
      v17 = v42;
LABEL_19:
      LODWORD(v42) = (2 * (v17 >> 1) + 2) | v17 & 1;
      if ( *v18 != -8 )
        --HIDWORD(v42);
      *v18 = v16;
      v22 = (unsigned int)v46;
      if ( (unsigned int)v46 >= HIDWORD(v46) )
      {
        v6 = (__int64)v47;
        sub_16CD150(&v45, v47, 0, 8);
        v22 = (unsigned int)v46;
      }
      v7 = v45;
      v45[v22] = v16;
      LODWORD(v46) = v46 + 1;
LABEL_10:
      ++v11;
    }
  }
  v3 = v2;
  result = sub_15ACB40(a1);
  if ( (_BYTE)result )
  {
    v5 = *a1;
    v39 = result;
    v43 = v1;
    v42 = 0x100000004LL;
    v41 = (const char *)&unk_49ECEC8;
    v44 = v3;
    sub_16027F0(v5, &v41);
    return v39;
  }
  return result;
}
