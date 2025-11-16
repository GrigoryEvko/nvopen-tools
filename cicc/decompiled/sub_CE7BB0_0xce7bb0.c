// Function: sub_CE7BB0
// Address: 0xce7bb0
//
__int64 __fastcall sub_CE7BB0(__int64 a1, char *a2, size_t a3, _QWORD *a4)
{
  size_t v5; // rdx
  __int64 v6; // r13
  char *v7; // r12
  char *v8; // rsi
  _BYTE *v9; // rdi
  __int64 v10; // r12
  unsigned int v11; // r13d
  __int64 v12; // r15
  unsigned __int8 v13; // al
  char **v14; // rdi
  __int64 v15; // r14
  unsigned int v16; // r12d
  __int64 v17; // rdx
  _BYTE *v18; // rdi
  const void *v19; // rax
  __int64 v20; // rdx
  unsigned __int8 v22; // al
  __int64 v23; // rax
  _BYTE *v24; // rbx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int64 v29; // [rsp+0h] [rbp-100h]
  int v33; // [rsp+30h] [rbp-D0h]
  unsigned int v34; // [rsp+34h] [rbp-CCh]
  __int64 v35; // [rsp+38h] [rbp-C8h]
  _QWORD *v36; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v37; // [rsp+48h] [rbp-B8h]
  _BYTE v38[176]; // [rsp+50h] [rbp-B0h] BYREF

  v5 = 0;
  v6 = *(_QWORD *)(a1 + 40);
  v36 = v38;
  v37 = 0x1000000000LL;
  v7 = off_4C5D0E8;
  if ( off_4C5D0E8 )
    v5 = strlen(off_4C5D0E8);
  v8 = v7;
  v9 = v36;
  v10 = sub_BA8DC0(v6, (__int64)v7, v5);
  if ( v10 )
  {
    v11 = 0;
    v33 = sub_B91A00(v10);
    if ( v33 )
    {
      while ( 1 )
      {
        v12 = sub_B91A10(v10, v11);
        v35 = v12 - 16;
        v13 = *(_BYTE *)(v12 - 16);
        v14 = (v13 & 2) != 0 ? *(char ***)(v12 - 32) : (char **)(v35 - 8LL * ((v13 >> 2) & 0xF));
        v8 = *v14;
        if ( *v14 )
        {
          if ( *v8 == 1 )
          {
            v8 = (char *)*((_QWORD *)v8 + 17);
            if ( (unsigned __int8)*v8 <= 3u && (char *)a1 == v8 )
            {
              if ( (v13 & 2) != 0 )
              {
                v34 = *(_DWORD *)(v12 - 24);
              }
              else
              {
                v8 = (char *)*(unsigned __int16 *)(v12 - 16);
                LOWORD(v8) = (unsigned __int16)v8 >> 6;
                v34 = (unsigned __int8)v8 & 0xF;
              }
              if ( v34 > 1 )
                break;
            }
          }
        }
LABEL_25:
        if ( v33 == ++v11 )
          goto LABEL_26;
      }
      v29 = v10;
      v15 = 8;
      v16 = 1;
      if ( (v13 & 2) != 0 )
      {
LABEL_15:
        v17 = *(_QWORD *)(v12 - 32);
        goto LABEL_16;
      }
      while ( 1 )
      {
        v17 = v35 - 8LL * ((v13 >> 2) & 0xF);
LABEL_16:
        v18 = *(_BYTE **)(v17 + v15);
        if ( *v18 )
          v18 = 0;
        v19 = (const void *)sub_B91420((__int64)v18);
        if ( a3 == v20 )
        {
          if ( !a3 )
            break;
          v8 = a2;
          if ( !memcmp(v19, a2, a3) )
            break;
        }
        v16 += 2;
        v15 += 16;
        if ( v34 <= v16 )
        {
          v10 = v29;
          goto LABEL_25;
        }
        v13 = *(_BYTE *)(v12 - 16);
        if ( (v13 & 2) != 0 )
          goto LABEL_15;
      }
      v22 = *(_BYTE *)(v12 - 16);
      if ( (v22 & 2) != 0 )
        v23 = *(_QWORD *)(v12 - 32);
      else
        v23 = v35 - 8LL * ((v22 >> 2) & 0xF);
      v24 = sub_CE7B90((_QWORD *)(v23 + 8LL * (v16 + 1)));
      v27 = (unsigned int)v37;
      v28 = (unsigned int)v37 + 1LL;
      if ( v28 > HIDWORD(v37) )
      {
        v8 = v38;
        sub_C8D5F0((__int64)&v36, v38, v28, 8u, v25, v26);
        v27 = (unsigned int)v37;
      }
      v36[v27] = v24;
      LODWORD(v37) = v37 + 1;
      v9 = v36;
      LODWORD(v10) = 1;
      *a4 = *v36;
    }
    else
    {
LABEL_26:
      v9 = v36;
      LODWORD(v10) = 0;
    }
  }
  if ( v9 != v38 )
    _libc_free(v9, v8);
  return (unsigned int)v10;
}
