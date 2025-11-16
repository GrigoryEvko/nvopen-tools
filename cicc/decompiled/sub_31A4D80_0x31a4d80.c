// Function: sub_31A4D80
// Address: 0x31a4d80
//
void __fastcall sub_31A4D80(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rcx
  __int64 v9; // r8
  char **v10; // r13
  char **v11; // r15
  __int64 *v12; // rbx
  unsigned __int8 v13; // dl
  bool v14; // cl
  char *v15; // r12
  char v16; // al
  unsigned __int64 v17; // rdx
  __int64 v18; // rsi
  char *v19; // r10
  char *v20; // rsi
  _BYTE *v21; // r11
  __int64 *v22; // rsi
  char *v23; // rbx
  unsigned int v24; // r12d
  char **v25; // rax
  char *v26; // r13
  char *v27; // rcx
  __int64 v28; // r8
  __int64 v29; // rdx
  unsigned __int64 v30; // r9
  char **v31; // [rsp+10h] [rbp-80h]
  _BYTE *v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+20h] [rbp-70h]
  __int64 *v34; // [rsp+30h] [rbp-60h] BYREF
  __int64 v35; // [rsp+38h] [rbp-58h]
  char v36; // [rsp+40h] [rbp-50h] BYREF

  v6 = sub_D49300(*(_QWORD *)(a1 + 104), a2, a3, a4, a5, a6);
  if ( v6 )
  {
    v7 = *(_BYTE *)(v6 - 16);
    if ( (v7 & 2) != 0 )
    {
      v9 = *(_QWORD *)(v6 - 32);
      v8 = *(unsigned int *)(v6 - 24);
    }
    else
    {
      v8 = (*(_WORD *)(v6 - 16) >> 6) & 0xF;
      v9 = v6 - 8LL * ((v7 >> 2) & 0xF) - 16;
    }
    v10 = (char **)(v9 + 8 * v8);
    v11 = (char **)(v9 + 8);
    v12 = (__int64 *)&v36;
    if ( v10 != (char **)(v9 + 8) )
    {
      while ( 1 )
      {
        v34 = v12;
        v35 = 0x400000000LL;
        v15 = *v11;
        v16 = **v11;
        if ( (unsigned __int8)(v16 - 5) > 0x1Fu )
        {
          if ( !v16 )
          {
LABEL_11:
            v18 = sub_B91420((__int64)v15);
            if ( (_DWORD)v35 == 1 )
              sub_31A4C60((const char **)a1, v18, v17, *v34);
            goto LABEL_13;
          }
          goto LABEL_8;
        }
        v13 = *(v15 - 16);
        v14 = (v13 & 2) != 0;
        if ( (v13 & 2) == 0 )
          break;
        if ( *((_DWORD *)v15 - 6) )
        {
          v20 = (char *)*((_QWORD *)v15 - 4);
          v19 = v15 - 16;
LABEL_19:
          v21 = *(_BYTE **)v20;
          v22 = v12;
          v23 = *v11;
          if ( *v21 )
            v21 = 0;
          v24 = 1;
          v25 = v10;
          v26 = v19;
          while ( v14 )
          {
            if ( v24 >= *((_DWORD *)v23 - 6) )
              goto LABEL_30;
            v27 = (char *)*((_QWORD *)v23 - 4);
LABEL_24:
            v28 = *(_QWORD *)&v27[8 * v24];
            v29 = (unsigned int)v35;
            v30 = (unsigned int)v35 + 1LL;
            if ( v30 > HIDWORD(v35) )
            {
              v31 = v25;
              v32 = v21;
              v33 = *(_QWORD *)&v27[8 * v24];
              sub_C8D5F0((__int64)&v34, v22, (unsigned int)v35 + 1LL, 8u, v28, v30);
              v29 = (unsigned int)v35;
              v25 = v31;
              v21 = v32;
              v28 = v33;
            }
            ++v24;
            v34[v29] = v28;
            LODWORD(v35) = v35 + 1;
            v13 = *(v23 - 16);
            v14 = (v13 & 2) != 0;
          }
          if ( v24 < ((*((_WORD *)v23 - 8) >> 6) & 0xFu) )
          {
            v27 = &v26[-8 * ((v13 >> 2) & 0xF)];
            goto LABEL_24;
          }
LABEL_30:
          v10 = v25;
          v12 = v22;
          if ( v21 )
          {
            v15 = v21;
            goto LABEL_11;
          }
LABEL_13:
          if ( v34 == v12 )
            goto LABEL_8;
          _libc_free((unsigned __int64)v34);
          if ( v10 == ++v11 )
            return;
        }
        else
        {
LABEL_8:
          if ( v10 == ++v11 )
            return;
        }
      }
      if ( (*((_WORD *)v15 - 8) & 0x3C0) == 0 )
        goto LABEL_8;
      v19 = v15 - 16;
      v20 = &v15[-8 * ((v13 >> 2) & 0xF) - 16];
      goto LABEL_19;
    }
  }
}
