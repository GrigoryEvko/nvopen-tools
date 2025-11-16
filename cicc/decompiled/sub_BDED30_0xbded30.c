// Function: sub_BDED30
// Address: 0xbded30
//
void __fastcall sub_BDED30(__int64 *a1, _BYTE *a2, const char *a3)
{
  __int64 v5; // rdi
  const char **v6; // rax
  __int64 v7; // rcx
  const char **v8; // rdx
  char v9; // dl
  bool v10; // al
  const char **v11; // rcx
  const char *v12; // r14
  const char *v13; // rdx
  const char *v14; // rcx
  __int64 *v15; // rcx
  unsigned __int8 *v16; // r14
  char v17; // dl
  unsigned __int8 *v18; // rax
  const char *v19; // r15
  char v20; // dl
  _BYTE *v21; // rax
  __int64 *v22; // rdi
  const char **v23; // rbx
  unsigned __int8 v24; // r8
  const char *v25; // rcx
  const char *v26; // rcx
  __int64 *v27; // r15
  _BYTE *v28; // rax
  const char **v29; // rbx
  __int64 *v30; // r13
  _BYTE *v31; // [rsp-78h] [rbp-78h]
  _BYTE *v32; // [rsp-70h] [rbp-70h]
  _QWORD v33[4]; // [rsp-68h] [rbp-68h] BYREF
  char v34; // [rsp-48h] [rbp-48h]
  char v35; // [rsp-47h] [rbp-47h]

  if ( !a3 || *a3 != 6 )
    return;
  v5 = *a1;
  if ( !*(_BYTE *)(v5 + 28) )
    goto LABEL_10;
  v6 = *(const char ***)(v5 + 8);
  v7 = *(unsigned int *)(v5 + 20);
  v8 = &v6[v7];
  if ( v6 == v8 )
  {
LABEL_9:
    if ( (unsigned int)v7 < *(_DWORD *)(v5 + 16) )
    {
      *(_DWORD *)(v5 + 20) = v7 + 1;
      *v8 = a3;
      ++*(_QWORD *)v5;
LABEL_11:
      v10 = (*(a3 - 16) & 2) != 0;
      if ( (*(a3 - 16) & 2) != 0 )
        v11 = (const char **)*((_QWORD *)a3 - 4);
      else
        v11 = (const char **)&a3[-8 * (((unsigned __int8)*(a3 - 16) >> 2) & 0xF) - 16];
      v12 = *v11;
      if ( *v11 && (v13 = a3, (unsigned __int8)(*v12 - 18) <= 2u) )
      {
        while ( 1 )
        {
          if ( v10 )
          {
            v14 = (const char *)*((_QWORD *)v13 - 4);
            if ( *((_DWORD *)v13 - 6) != 2 )
              goto LABEL_17;
          }
          else
          {
            v24 = *(v13 - 16);
            v25 = v13 - 16;
            if ( ((*((_WORD *)v13 - 8) >> 6) & 0xF) != 2 )
              goto LABEL_47;
            v14 = &v25[-8 * (((unsigned __int8)*(v13 - 16) >> 2) & 0xF)];
          }
          v26 = (const char *)*((_QWORD *)v14 + 1);
          if ( !v26 )
            break;
          v13 = v26;
          v10 = (*(v26 - 16) & 2) != 0;
        }
        v25 = v13 - 16;
        if ( v10 )
        {
LABEL_17:
          v15 = (__int64 *)*((_QWORD *)v13 - 4);
          goto LABEL_18;
        }
        v24 = *(v13 - 16);
LABEL_47:
        v15 = (__int64 *)&v25[-8 * ((v24 >> 2) & 0xF)];
LABEL_18:
        v16 = (unsigned __int8 *)*v15;
        if ( *v15 )
        {
          sub_AE6EC0(*a1, *v15);
          if ( v17 )
          {
            v18 = sub_AF34D0(v16);
            v19 = (const char *)v18;
            if ( !v18 || v18 == v16 || (sub_AE6EC0(*a1, (__int64)v18), v20) )
            {
              if ( !sub_AF3E00((__int64)v19, a1[2]) )
              {
                v21 = (_BYTE *)a1[2];
                v22 = (__int64 *)a1[3];
                v35 = 1;
                v34 = 3;
                v23 = (const char **)a1[1];
                v31 = v21;
                v33[0] = "!dbg attachment points at wrong subprogram for function";
                sub_BDD6D0(v22, (__int64)v33);
                if ( *v22 )
                {
                  if ( *v23 )
                    sub_BD9900(v22, *v23);
                  if ( v31 )
                    sub_BDBD80((__int64)v22, v31);
                  sub_BDBD80((__int64)v22, a2);
                  sub_BD9900(v22, a3);
                  sub_BD9900(v22, (const char *)v16);
                  if ( v19 )
                    sub_BD9900(v22, v19);
                }
              }
            }
          }
        }
        else
        {
          v30 = (__int64 *)a1[3];
          v35 = 1;
          v33[0] = "Failed to find DILocalScope";
          v34 = 3;
          sub_BDBF70(v30, (__int64)v33);
          if ( *v30 )
            sub_BD9900(v30, a3);
        }
      }
      else
      {
        v27 = (__int64 *)a1[3];
        v28 = (_BYTE *)a1[2];
        v35 = 1;
        v34 = 3;
        v29 = (const char **)a1[1];
        v32 = v28;
        v33[0] = "DILocation's scope must be a DILocalScope";
        sub_BDD6D0(v27, (__int64)v33);
        if ( *v27 )
        {
          if ( *v29 )
            sub_BD9900(v27, *v29);
          if ( v32 )
            sub_BDBD80((__int64)v27, v32);
          sub_BDBD80((__int64)v27, a2);
          sub_BD9900(v27, a3);
          if ( v12 )
            sub_BD9900(v27, v12);
        }
      }
      return;
    }
LABEL_10:
    sub_C8CC70(v5, a3);
    if ( !v9 )
      return;
    goto LABEL_11;
  }
  while ( a3 != *v6 )
  {
    if ( v8 == ++v6 )
      goto LABEL_9;
  }
}
