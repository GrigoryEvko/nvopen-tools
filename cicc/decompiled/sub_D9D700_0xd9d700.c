// Function: sub_D9D700
// Address: 0xd9d700
//
char __fastcall sub_D9D700(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r9
  _QWORD *v5; // rdx
  int v6; // ecx
  unsigned int v7; // r15d
  unsigned int v8; // r10d
  __int64 *v9; // rsi
  __int64 v10; // rdi
  __int64 *v11; // rdi
  int v12; // ecx
  char v13; // di
  int v14; // ecx
  unsigned int v15; // r8d
  __int64 *v16; // r15
  __int64 v17; // rax
  __int64 *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // r10
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r12
  int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r15
  __int64 *v31; // r12
  __int64 *v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  unsigned __int64 v35; // rdx
  unsigned int v36; // r11d
  int v37; // r10d
  unsigned int v38; // r11d
  char v40; // [rsp+17h] [rbp-E9h]
  __int64 v41; // [rsp+18h] [rbp-E8h]
  __int64 v42; // [rsp+18h] [rbp-E8h]
  _QWORD *v43; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v44; // [rsp+28h] [rbp-D8h]
  _QWORD v45[8]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v46; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v47; // [rsp+78h] [rbp-88h]
  __int64 v48; // [rsp+80h] [rbp-80h]
  int v49; // [rsp+88h] [rbp-78h]
  char v50; // [rsp+8Ch] [rbp-74h]
  __int64 v51; // [rsp+90h] [rbp-70h] BYREF

  if ( !a2 )
  {
    sub_D9D4C0(a1 + 904, 0);
    LOBYTE(v3) = sub_D9D270(a1 + 840, 0);
    return v3;
  }
  LOBYTE(v3) = sub_D97040(a1, *(_QWORD *)(a2 + 8));
  v40 = v3;
  if ( (_BYTE)v3 )
  {
    v3 = sub_D98300(a1, a2);
    if ( v3 )
    {
      v45[0] = v3;
      v44 = 0x800000001LL;
      v48 = 0x100000008LL;
      v47 = &v51;
      v5 = v45;
      v51 = v3;
      LODWORD(v3) = 1;
      v43 = v45;
      v49 = 0;
      v50 = 1;
      v46 = 1;
      while ( 1 )
      {
        v23 = (unsigned int)v3;
        LODWORD(v3) = v3 - 1;
        v24 = *(_QWORD *)(a1 + 848);
        v25 = v5[v23 - 1];
        v26 = *(_DWORD *)(a1 + 864);
        LODWORD(v44) = v3;
        if ( v26 )
        {
          v6 = v26 - 1;
          v7 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
          v8 = v6 & v7;
          v9 = (__int64 *)(v24 + 40LL * (v6 & v7));
          v10 = *v9;
          if ( v25 == *v9 )
          {
LABEL_7:
            v11 = (__int64 *)v9[1];
            if ( v11 != v9 + 3 )
              _libc_free(v11, v9);
            *v9 = -8192;
            v12 = *(_DWORD *)(a1 + 928);
            --*(_DWORD *)(a1 + 856);
            a2 = *(_QWORD *)(a1 + 912);
            ++*(_DWORD *)(a1 + 860);
            if ( v12 )
            {
              v13 = v40;
              goto LABEL_11;
            }
LABEL_15:
            v19 = *(unsigned int *)(a1 + 960);
            v20 = *(_QWORD *)(a1 + 944);
            if ( (_DWORD)v19 )
            {
              v21 = ((_DWORD)v19 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              a2 = v20 + 104 * v21;
              v22 = *(_QWORD *)a2;
              if ( *(_QWORD *)a2 == v25 )
              {
LABEL_17:
                if ( a2 != v20 + 104 * v19 )
                {
                  v27 = *(__int64 **)(a2 + 16);
                  v28 = *(_BYTE *)(a2 + 36) ? *(unsigned int *)(a2 + 28) : *(unsigned int *)(a2 + 24);
                  v29 = (__int64)&v27[v28];
                  if ( v27 != (__int64 *)v29 )
                  {
                    while ( 1 )
                    {
                      v30 = *v27;
                      v31 = v27;
                      if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                      if ( (__int64 *)v29 == ++v27 )
                        goto LABEL_18;
                    }
LABEL_30:
                    if ( (__int64 *)v29 == v31 )
                      goto LABEL_18;
                    if ( !v50 )
                      goto LABEL_46;
                    v32 = v47;
                    v21 = HIDWORD(v48);
                    v28 = (__int64)&v47[HIDWORD(v48)];
                    if ( v47 != (__int64 *)v28 )
                    {
                      while ( v30 != *v32 )
                      {
                        if ( (__int64 *)v28 == ++v32 )
                          goto LABEL_50;
                      }
                      goto LABEL_36;
                    }
LABEL_50:
                    if ( HIDWORD(v48) < (unsigned int)v48 )
                    {
                      ++HIDWORD(v48);
                      *(_QWORD *)v28 = v30;
                      ++v46;
LABEL_47:
                      v34 = (unsigned int)v44;
                      v21 = HIDWORD(v44);
                      v35 = (unsigned int)v44 + 1LL;
                      if ( v35 > HIDWORD(v44) )
                      {
                        a2 = (__int64)v45;
                        v42 = v29;
                        sub_C8D5F0((__int64)&v43, v45, v35, 8u, v29, v4);
                        v34 = (unsigned int)v44;
                        v29 = v42;
                      }
                      v28 = (__int64)v43;
                      v43[v34] = v30;
                      LODWORD(v44) = v44 + 1;
                    }
                    else
                    {
LABEL_46:
                      a2 = v30;
                      v41 = v29;
                      sub_C8CC70((__int64)&v46, v30, v28, v21, v29, v4);
                      v29 = v41;
                      if ( (_BYTE)v28 )
                        goto LABEL_47;
                    }
LABEL_36:
                    v33 = v31 + 1;
                    if ( v31 + 1 == (__int64 *)v29 )
                      goto LABEL_18;
                    while ( 1 )
                    {
                      v30 = *v33;
                      v31 = v33;
                      if ( (unsigned __int64)*v33 < 0xFFFFFFFFFFFFFFFELL )
                        goto LABEL_30;
                      if ( (__int64 *)v29 == ++v33 )
                        goto LABEL_18;
                    }
                  }
                }
              }
              else
              {
                a2 = 1;
                while ( v22 != -4096 )
                {
                  v38 = a2 + 1;
                  v21 = ((_DWORD)v19 - 1) & (unsigned int)(a2 + v21);
                  a2 = v20 + 104LL * (unsigned int)v21;
                  v22 = *(_QWORD *)a2;
                  if ( v25 == *(_QWORD *)a2 )
                    goto LABEL_17;
                  a2 = v38;
                }
              }
            }
            goto LABEL_18;
          }
          a2 = 1;
          while ( v10 != -4096 )
          {
            v36 = a2 + 1;
            v8 = v6 & (a2 + v8);
            v9 = (__int64 *)(v24 + 40LL * v8);
            v10 = *v9;
            if ( v25 == *v9 )
              goto LABEL_7;
            a2 = v36;
          }
        }
        v12 = *(_DWORD *)(a1 + 928);
        if ( v12 )
        {
          a2 = *(_QWORD *)(a1 + 912);
          v13 = 0;
          v7 = ((unsigned int)v25 >> 4) ^ ((unsigned int)v25 >> 9);
LABEL_11:
          v14 = v12 - 1;
          v15 = v14 & v7;
          v16 = (__int64 *)(a2 + 40LL * (v14 & v7));
          v17 = *v16;
          if ( v25 == *v16 )
          {
LABEL_12:
            v18 = (__int64 *)v16[1];
            if ( v18 != v16 + 3 )
              _libc_free(v18, a2);
            *v16 = -8192;
            --*(_DWORD *)(a1 + 920);
            ++*(_DWORD *)(a1 + 924);
            goto LABEL_15;
          }
          v37 = 1;
          while ( v17 != -4096 )
          {
            v15 = v14 & (v37 + v15);
            v16 = (__int64 *)(a2 + 40LL * v15);
            v17 = *v16;
            if ( v25 == *v16 )
              goto LABEL_12;
            ++v37;
          }
          if ( v13 )
            goto LABEL_15;
LABEL_18:
          LODWORD(v3) = v44;
        }
        if ( !(_DWORD)v3 )
        {
          if ( !v50 )
            LOBYTE(v3) = _libc_free(v47, a2);
          if ( v43 != v45 )
            LOBYTE(v3) = _libc_free(v43, a2);
          return v3;
        }
        v5 = v43;
      }
    }
  }
  return v3;
}
