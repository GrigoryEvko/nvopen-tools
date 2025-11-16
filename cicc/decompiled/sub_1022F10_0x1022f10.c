// Function: sub_1022F10
// Address: 0x1022f10
//
_QWORD *__fastcall sub_1022F10(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  int v6; // edi
  int v7; // eax
  unsigned int v8; // r10d
  int v9; // r13d
  unsigned __int8 *v10; // rdi
  int v11; // eax
  unsigned __int8 *v12; // r9
  _QWORD *v13; // r11
  unsigned __int8 **v15; // rcx
  unsigned __int8 *v16; // rax
  __int64 v17; // rdx
  char v18; // al
  char v19; // al
  __int64 v20; // r8
  unsigned int v21; // r10d
  __int64 v22; // r9
  __int64 v23; // rax
  unsigned __int8 *v24; // rbx
  bool v25; // r14
  __int64 v26; // rcx
  unsigned __int64 v27; // rdx
  bool v28; // zf
  unsigned __int64 v29; // rax
  int v30; // edx
  unsigned int v31; // r12d
  unsigned __int8 *v32; // r15
  int v33; // eax
  __int64 v34; // r9
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  int v39; // edx
  __int64 v40; // rdx
  _QWORD *v41; // [rsp+8h] [rbp-B8h]
  unsigned int v42; // [rsp+8h] [rbp-B8h]
  unsigned __int8 *v43; // [rsp+10h] [rbp-B0h]
  unsigned __int8 *v44; // [rsp+10h] [rbp-B0h]
  unsigned int v45; // [rsp+18h] [rbp-A8h]
  _QWORD *v46; // [rsp+18h] [rbp-A8h]
  _QWORD *v47; // [rsp+18h] [rbp-A8h]
  int v48; // [rsp+20h] [rbp-A0h]
  _QWORD *v49; // [rsp+20h] [rbp-A0h]
  unsigned int v50; // [rsp+28h] [rbp-98h]
  __int64 v51; // [rsp+28h] [rbp-98h]
  __int64 v52; // [rsp+30h] [rbp-90h] BYREF
  _DWORD v53[6]; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v54; // [rsp+50h] [rbp-70h] BYREF
  int v55; // [rsp+58h] [rbp-68h]
  char *v56; // [rsp+60h] [rbp-60h] BYREF
  __int64 v57; // [rsp+68h] [rbp-58h]
  _BYTE v58[80]; // [rsp+70h] [rbp-50h] BYREF

  v5 = a2;
  v6 = *(_DWORD *)(a2 + 40);
  v56 = v58;
  v57 = 0x400000000LL;
  v7 = sub_1022EF0(v6);
  v8 = v7 - 53;
  v9 = v7;
  if ( (unsigned int)(v7 - 53) <= 1 )
  {
    v10 = *(unsigned __int8 **)(a2 + 32);
    v12 = v10;
    if ( *v10 != 84 )
    {
      v48 = 0;
      v13 = a1 + 2;
      v50 = 2;
      goto LABEL_42;
    }
    v50 = 2;
  }
  else
  {
    v10 = *(unsigned __int8 **)(a2 + 32);
    v11 = *v10;
    v12 = v10;
    if ( (_BYTE)v11 != 84 )
    {
      v48 = 0;
      v13 = a1 + 2;
      v50 = 1;
      goto LABEL_4;
    }
    v50 = 1;
  }
  v13 = a1 + 2;
  if ( (*((_DWORD *)v10 + 1) & 0x7FFFFFF) != 2 )
    goto LABEL_6;
  v15 = (unsigned __int8 **)*((_QWORD *)v10 - 1);
  v16 = *v15;
  v12 = v15[4];
  if ( **v15 <= 0x1Cu )
    v16 = 0;
  if ( *v12 <= 0x1Cu )
    v12 = 0;
  if ( (unsigned __int8 *)a3 == v16 )
    goto LABEL_19;
  if ( (unsigned __int8 *)a3 != v12 )
  {
LABEL_6:
    *a1 = v13;
    a1[1] = 0x400000000LL;
    goto LABEL_7;
  }
  v12 = v16;
LABEL_19:
  if ( v8 > 1 )
  {
    v48 = 1;
    v11 = *v12;
LABEL_4:
    if ( ((_BYTE)v11 != 85
       || (v17 = *((_QWORD *)v12 - 4)) == 0
       || *(_BYTE *)v17
       || *(_QWORD *)(v17 + 24) != *((_QWORD *)v12 + 10)
       || (*(_BYTE *)(v17 + 33) & 0x20) == 0
       || *(_DWORD *)(v17 + 36) != 174)
      && v9 != v11 - 29 )
    {
      goto LABEL_6;
    }
    goto LABEL_28;
  }
  v48 = 1;
LABEL_42:
  a2 = (__int64)v53;
  v46 = v13;
  v42 = v8;
  v44 = v12;
  v29 = sub_99AEC0(v12, (__int64 *)v53, (__int64 *)&v54, 0, 0);
  v13 = v46;
  *(_QWORD *)&v53[3] = v29;
  v53[5] = v30;
  if ( (unsigned int)(v29 - 7) <= 1 || !(_DWORD)v29 )
    goto LABEL_6;
  v10 = *(unsigned __int8 **)(v5 + 32);
  v12 = v44;
  v8 = v42;
LABEL_28:
  a2 = 2;
  v41 = v13;
  v43 = v12;
  v45 = v8;
  v18 = sub_BD3610((__int64)v10, 2);
  v13 = v41;
  if ( !v18 )
    goto LABEL_6;
  a2 = v50 + v48;
  v19 = sub_BD3610(a3, a2);
  v21 = v45;
  v22 = (__int64)v43;
  v13 = v41;
  if ( !v19 )
    goto LABEL_6;
  v23 = *(_QWORD *)(a3 + 16);
  if ( !v23 )
  {
    if ( !v43 )
      goto LABEL_36;
    goto LABEL_6;
  }
  do
  {
    v24 = *(unsigned __int8 **)(v23 + 24);
    if ( *v24 != 84 )
    {
      v25 = v45 > 1;
      if ( *v24 == 86 || v45 > 1 )
      {
        if ( v24 == v43 )
          goto LABEL_36;
        v47 = a1;
        v31 = v21;
        v32 = *(unsigned __int8 **)(v23 + 24);
        while ( 2 )
        {
          if ( v31 <= 1 )
          {
            a2 = (__int64)&v52;
            v38 = sub_99AEC0(v32, &v52, (__int64 *)v53, 0, 0);
            v54 = v38;
            v55 = v39;
            if ( (unsigned int)(v38 - 7) <= 1 || !(_DWORD)v38 )
              break;
          }
          else
          {
            v33 = *v32;
            if ( ((_BYTE)v33 != 85
               || (v40 = *((_QWORD *)v32 - 4)) == 0
               || *(_BYTE *)v40
               || *(_QWORD *)(v40 + 24) != *((_QWORD *)v32 + 10)
               || (*(_BYTE *)(v40 + 33) & 0x20) == 0
               || *(_DWORD *)(v40 + 36) != 174)
              && v9 != v33 - 29 )
            {
              break;
            }
          }
          a2 = v50;
          if ( (unsigned __int8)sub_BD3610((__int64)v32, v50) )
          {
            v35 = (unsigned int)v57;
            v36 = (unsigned int)v57 + 1LL;
            if ( v36 > HIDWORD(v57) )
            {
              a2 = (__int64)v58;
              sub_C8D5F0((__int64)&v56, v58, v36, 8u, v20, v34);
              v35 = (unsigned int)v57;
            }
            *(_QWORD *)&v56[8 * v35] = v32;
            v26 = (unsigned int)(v57 + 1);
            v37 = *((_QWORD *)v32 + 2);
            LODWORD(v57) = v57 + 1;
            if ( v37 )
            {
              while ( 1 )
              {
                v32 = *(unsigned __int8 **)(v37 + 24);
                if ( *v32 != 84 && (*v32 == 86 || v25) )
                  break;
                v37 = *(_QWORD *)(v37 + 8);
                if ( !v37 )
                {
                  a1 = v47;
                  v13 = v41;
                  v22 = (__int64)v43;
                  if ( !v43 )
                    goto LABEL_37;
                  goto LABEL_6;
                }
              }
              if ( v32 != v43 )
                continue;
              a1 = v47;
              v13 = v41;
              v22 = (__int64)v43;
              goto LABEL_37;
            }
            a1 = v47;
            v13 = v41;
            v22 = (__int64)v43;
            if ( !v43 )
              goto LABEL_37;
            goto LABEL_6;
          }
          break;
        }
        a1 = v47;
        v13 = v41;
        goto LABEL_6;
      }
    }
    v23 = *(_QWORD *)(v23 + 8);
  }
  while ( v23 );
  if ( v43 )
    goto LABEL_6;
LABEL_36:
  v26 = (unsigned int)v57;
LABEL_37:
  v27 = v26 + 1;
  if ( v26 + 1 > (unsigned __int64)HIDWORD(v57) )
  {
    a2 = (__int64)v58;
    v49 = v13;
    v51 = v22;
    sub_C8D5F0((__int64)&v56, v58, v27, 8u, v20, v22);
    v26 = (unsigned int)v57;
    v13 = v49;
    v22 = v51;
  }
  *(_QWORD *)&v56[8 * v26] = v22;
  v28 = (_DWORD)v57 == -1;
  LODWORD(v57) = v57 + 1;
  *a1 = v13;
  a1[1] = 0x400000000LL;
  if ( !v28 )
  {
    a2 = (__int64)&v56;
    sub_1021AD0((__int64)a1, &v56, v27, 0x400000000LL, v20, v22);
  }
LABEL_7:
  if ( v56 != v58 )
    _libc_free(v56, a2);
  return a1;
}
