// Function: sub_11560B0
// Address: 0x11560b0
//
_QWORD *__fastcall sub_11560B0(unsigned __int8 *a1, __int64 a2)
{
  char *v3; // rbx
  char *v4; // r15
  __int64 v5; // r13
  unsigned __int8 v6; // al
  unsigned int v7; // r14d
  unsigned __int8 v8; // dl
  __int64 v9; // rsi
  _QWORD *v10; // r12
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // rcx
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 *v17; // r12
  __int64 v18; // r15
  __int64 v19; // rdx
  int v20; // r14d
  __int64 v21; // rbx
  __int64 v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rcx
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 *v28; // r12
  _QWORD *v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // r14
  _QWORD *v32; // rax
  __int64 v33; // rdx
  int v34; // r14d
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rdx
  unsigned int v38; // esi
  __int64 v39; // [rsp+0h] [rbp-A0h]
  __int64 v40; // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+8h] [rbp-98h]
  int v42[8]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v43; // [rsp+30h] [rbp-70h]
  _BYTE v44[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v45; // [rsp+60h] [rbp-40h]

  v3 = (char *)*((_QWORD *)a1 - 8);
  v4 = (char *)*((_QWORD *)a1 - 4);
  v5 = *((_QWORD *)a1 + 1);
  v6 = *v3;
  v7 = *a1 - 29;
  v8 = *v4;
  if ( (unsigned __int8)*v3 <= 0x1Cu )
    goto LABEL_5;
  v9 = *((_QWORD *)v3 + 2);
  if ( v6 != 68 || (v12 = *((_QWORD *)v3 - 4)) == 0 || v8 != 68 )
  {
    if ( v9 && !*(_QWORD *)(v9 + 8) && v6 == 68 )
    {
      v12 = *((_QWORD *)v3 - 4);
LABEL_34:
      if ( v12 && v8 <= 0x15u )
      {
        v41 = v12;
        v26 = sub_AD4C30(*((_QWORD *)a1 - 4), *(__int64 ***)(v12 + 8), 0);
        v27 = sub_96F480(0x27u, v26, *((_QWORD *)v4 + 1), *(_QWORD *)(a2 + 88));
        if ( v4 == (char *)v27 && v27 != 0 && v26 )
        {
          v28 = *(__int64 **)(a2 + 32);
          v43 = 257;
          v18 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v28[10] + 16LL))(
                  v28[10],
                  v7,
                  v41,
                  v26);
          if ( !v18 )
          {
            v45 = 257;
            v18 = sub_B504D0(v7, v41, v26, (__int64)v44, 0, 0);
            if ( (unsigned __int8)sub_920620(v18) )
            {
              v33 = v28[12];
              v34 = *((_DWORD *)v28 + 26);
              if ( v33 )
                sub_B99FD0(v18, 3u, v33);
              sub_B45150(v18, v34);
            }
            (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v28[11] + 16LL))(
              v28[11],
              v18,
              v42,
              v28[7],
              v28[8]);
            v35 = *v28;
            v36 = *v28 + 16LL * *((unsigned int *)v28 + 2);
            while ( v36 != v35 )
            {
              v37 = *(_QWORD *)(v35 + 8);
              v38 = *(_DWORD *)v35;
              v35 += 16;
              sub_B99FD0(v18, v38, v37);
            }
          }
          goto LABEL_39;
        }
        return 0;
      }
    }
LABEL_5:
    if ( v8 <= 0x1Cu )
      return 0;
    goto LABEL_13;
  }
  v13 = *((_QWORD *)v4 - 4);
  if ( !v13 || *(_QWORD *)(v13 + 8) != *(_QWORD *)(v12 + 8) )
  {
LABEL_12:
    if ( !v9 )
    {
LABEL_13:
      v14 = *((_QWORD *)v4 + 2);
      if ( v14 )
      {
        if ( !*(_QWORD *)(v14 + 8) && v8 == 68 )
        {
          v15 = *((_QWORD *)v4 - 4);
          if ( v15 )
          {
            if ( v6 <= 0x15u )
            {
              v39 = *((_QWORD *)v4 - 4);
              v40 = sub_AD4C30(*((_QWORD *)a1 - 8), *(__int64 ***)(v15 + 8), 0);
              v16 = sub_96F480(0x27u, v40, *((_QWORD *)v3 + 1), *(_QWORD *)(a2 + 88));
              if ( v3 == (char *)v16 && v16 != 0 )
              {
                if ( v40 )
                {
                  v17 = *(__int64 **)(a2 + 32);
                  v43 = 257;
                  v18 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v17[10] + 16LL))(
                          v17[10],
                          v7,
                          v40,
                          v39);
                  if ( !v18 )
                  {
                    v45 = 257;
                    v18 = sub_B504D0(v7, v40, v39, (__int64)v44, 0, 0);
                    if ( (unsigned __int8)sub_920620(v18) )
                    {
                      v19 = v17[12];
                      v20 = *((_DWORD *)v17 + 26);
                      if ( v19 )
                        sub_B99FD0(v18, 3u, v19);
                      sub_B45150(v18, v20);
                    }
                    (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v17[11] + 16LL))(
                      v17[11],
                      v18,
                      v42,
                      v17[7],
                      v17[8]);
                    v21 = *v17;
                    v22 = *v17 + 16LL * *((unsigned int *)v17 + 2);
                    while ( v22 != v21 )
                    {
                      v23 = *(_QWORD *)(v21 + 8);
                      v24 = *(_DWORD *)v21;
                      v21 += 16;
                      sub_B99FD0(v18, v24, v23);
                    }
                  }
LABEL_39:
                  v45 = 257;
                  v29 = sub_BD2C40(72, unk_3F10A14);
                  v10 = v29;
                  if ( v29 )
                    sub_B515B0((__int64)v29, v18, v5, (__int64)v44, 0, 0);
                  return v10;
                }
              }
            }
          }
        }
      }
      return 0;
    }
    goto LABEL_30;
  }
  if ( v9 )
  {
    if ( !*(_QWORD *)(v9 + 8) )
      goto LABEL_44;
    v25 = *((_QWORD *)v4 + 2);
    if ( !v25 )
    {
LABEL_30:
      if ( *(_QWORD *)(v9 + 8) )
        goto LABEL_13;
      goto LABEL_34;
    }
  }
  else
  {
    v25 = *((_QWORD *)v4 + 2);
    if ( !v25 )
      return 0;
  }
  if ( *(_QWORD *)(v25 + 8) )
    goto LABEL_12;
LABEL_44:
  v30 = *(__int64 **)(a2 + 32);
  v45 = 257;
  v31 = sub_1155FA0(v30, v7, v12, v13, v42[0], 0, (__int64)v44, 0);
  v45 = 257;
  v32 = sub_BD2C40(72, unk_3F10A14);
  v10 = v32;
  if ( v32 )
    sub_B515B0((__int64)v32, v31, v5, (__int64)v44, 0, 0);
  return v10;
}
