// Function: sub_164F440
// Address: 0x164f440
//
void __fastcall sub_164F440(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  int v5; // r14d
  unsigned int i; // r15d
  __int64 v7; // rsi
  __int64 *v8; // rax
  __int64 *v9; // rdi
  __int64 *v10; // rcx
  unsigned __int8 **v11; // rdi
  unsigned __int8 **v12; // rcx
  __int64 v13; // rax
  unsigned __int8 **v14; // r15
  unsigned __int8 **v15; // rax
  unsigned __int8 *v16; // r14
  unsigned __int8 **v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int8 **v19; // rax
  unsigned __int8 **v20; // r8
  unsigned __int8 **v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // r13
  _BYTE *v25; // rax
  __int64 v26; // rax
  char v27; // dl
  char v28; // al
  unsigned __int8 **v29; // rcx
  __int64 v30; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v31; // [rsp+8h] [rbp-98h]
  const char *v32; // [rsp+10h] [rbp-90h] BYREF
  char v33; // [rsp+20h] [rbp-80h]
  char v34; // [rsp+21h] [rbp-7Fh]
  const char *v35; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v36; // [rsp+38h] [rbp-68h]
  __int64 *v37; // [rsp+40h] [rbp-60h]
  __int64 v38; // [rsp+48h] [rbp-58h]
  int v39; // [rsp+50h] [rbp-50h]
  _BYTE v40[72]; // [rsp+58h] [rbp-48h] BYREF

  if ( !(unsigned __int8)sub_16033B0(**(_QWORD **)(a1 + 8)) )
  {
    v2 = *(_QWORD *)(a1 + 8);
    v35 = "llvm.dbg.cu";
    LOWORD(v37) = 259;
    v3 = sub_1632310(v2, (__int64)&v35);
    v35 = 0;
    v4 = v3;
    v38 = 2;
    v36 = (__int64 *)v40;
    v37 = (__int64 *)v40;
    v39 = 0;
    if ( v3 )
    {
      v5 = sub_161F520(v3);
      if ( v5 )
      {
        for ( i = 0; v5 != i; ++i )
        {
LABEL_8:
          v7 = sub_161F530(v4, i);
          v8 = v36;
          if ( v37 != v36 )
            goto LABEL_6;
          v9 = &v36[HIDWORD(v38)];
          if ( v36 != v9 )
          {
            v10 = 0;
            while ( v7 != *v8 )
            {
              if ( *v8 == -2 )
                v10 = v8;
              if ( v9 == ++v8 )
              {
                if ( !v10 )
                  goto LABEL_63;
                ++i;
                *v10 = v7;
                --v39;
                ++v35;
                if ( v5 != i )
                  goto LABEL_8;
                goto LABEL_17;
              }
            }
            continue;
          }
LABEL_63:
          if ( HIDWORD(v38) < (unsigned int)v38 )
          {
            ++HIDWORD(v38);
            *v9 = v7;
            ++v35;
          }
          else
          {
LABEL_6:
            sub_16CCBA0(&v35, v7);
          }
        }
      }
    }
LABEL_17:
    v11 = *(unsigned __int8 ***)(a1 + 672);
    v12 = *(unsigned __int8 ***)(a1 + 664);
    if ( v11 == v12 )
      v13 = *(unsigned int *)(a1 + 684);
    else
      v13 = *(unsigned int *)(a1 + 680);
    v14 = &v11[v13];
    if ( v11 == v14 )
    {
LABEL_23:
      v30 = a1 + 656;
    }
    else
    {
      v15 = *(unsigned __int8 ***)(a1 + 672);
      while ( 1 )
      {
        v16 = *v15;
        v17 = v15;
        if ( (unsigned __int64)*v15 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v14 == ++v15 )
          goto LABEL_23;
      }
      v30 = a1 + 656;
      if ( v14 != v15 )
      {
        v18 = (unsigned __int64)v37;
        v19 = (unsigned __int8 **)v36;
        if ( v37 == v36 )
          goto LABEL_45;
LABEL_26:
        v31 = v18 + 8LL * (unsigned int)v38;
        v19 = (unsigned __int8 **)sub_16CC9F0(&v35, v16);
        v20 = (unsigned __int8 **)v31;
        if ( *v19 == v16 )
        {
          v18 = (unsigned __int64)v37;
          if ( v37 == v36 )
            v29 = (unsigned __int8 **)&v37[HIDWORD(v38)];
          else
            v29 = (unsigned __int8 **)&v37[(unsigned int)v38];
          goto LABEL_52;
        }
        v18 = (unsigned __int64)v37;
        if ( v37 == v36 )
        {
          v19 = (unsigned __int8 **)&v37[HIDWORD(v38)];
          v29 = v19;
          goto LABEL_52;
        }
        v19 = (unsigned __int8 **)&v37[(unsigned int)v38];
LABEL_29:
        if ( v20 == v19 )
        {
LABEL_54:
          v24 = *(_QWORD *)a1;
          v34 = 1;
          v32 = "DICompileUnit not listed in llvm.dbg.cu";
          v33 = 3;
          if ( !v24 )
          {
            v28 = *(_BYTE *)(a1 + 74);
            *(_BYTE *)(a1 + 73) = 1;
            *(_BYTE *)(a1 + 72) |= v28;
LABEL_41:
            if ( (__int64 *)v18 != v36 )
              _libc_free(v18);
            return;
          }
          sub_16E2CE0(&v32, v24);
          v25 = *(_BYTE **)(v24 + 24);
          if ( (unsigned __int64)v25 >= *(_QWORD *)(v24 + 16) )
          {
            sub_16E7DE0(v24, 10);
          }
          else
          {
            *(_QWORD *)(v24 + 24) = v25 + 1;
            *v25 = 10;
          }
          v26 = *(_QWORD *)a1;
          v27 = *(_BYTE *)(a1 + 74);
          *(_BYTE *)(a1 + 73) = 1;
          *(_BYTE *)(a1 + 72) |= v27;
          if ( v16 && v26 )
            sub_164ED40((__int64 *)a1, v16);
          goto LABEL_40;
        }
        while ( 1 )
        {
          v21 = v17 + 1;
          if ( v17 + 1 == v14 )
            break;
          while ( 1 )
          {
            v16 = *v21;
            v17 = v21;
            if ( (unsigned __int64)*v21 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v14 == ++v21 )
              goto LABEL_33;
          }
          if ( v14 == v21 )
            break;
          v19 = (unsigned __int8 **)v36;
          if ( (__int64 *)v18 != v36 )
            goto LABEL_26;
LABEL_45:
          v20 = (unsigned __int8 **)(v18 + 8LL * HIDWORD(v38));
          if ( (unsigned __int8 **)v18 == v20 )
          {
            v29 = (unsigned __int8 **)v18;
          }
          else
          {
            do
            {
              if ( *v19 == v16 )
                break;
              ++v19;
            }
            while ( v20 != v19 );
            v29 = (unsigned __int8 **)(v18 + 8LL * HIDWORD(v38));
          }
LABEL_52:
          while ( v29 != v19 )
          {
            if ( (unsigned __int64)*v19 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_29;
            ++v19;
          }
          if ( v20 == v19 )
            goto LABEL_54;
        }
LABEL_33:
        v11 = *(unsigned __int8 ***)(a1 + 672);
        v12 = *(unsigned __int8 ***)(a1 + 664);
      }
    }
    ++*(_QWORD *)(a1 + 656);
    if ( v12 != v11 )
    {
      v22 = 4 * (*(_DWORD *)(a1 + 684) - *(_DWORD *)(a1 + 688));
      v23 = *(unsigned int *)(a1 + 680);
      if ( v22 < 0x20 )
        v22 = 32;
      if ( (unsigned int)v23 > v22 )
      {
        sub_16CC920(v30);
        goto LABEL_40;
      }
      memset(v11, -1, 8 * v23);
    }
    *(_QWORD *)(a1 + 684) = 0;
LABEL_40:
    v18 = (unsigned __int64)v37;
    goto LABEL_41;
  }
}
