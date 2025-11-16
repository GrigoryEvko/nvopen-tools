// Function: sub_349CBF0
// Address: 0x349cbf0
//
void __fastcall sub_349CBF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  _BYTE *v9; // rsi
  __int64 v10; // rbx
  _BYTE *v11; // rcx
  __int64 v12; // rdx
  void (__fastcall *v13)(__int64, __int64, __int64, _QWORD, _QWORD, unsigned __int64 *, __int64); // r11
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  unsigned __int64 *v17; // r12
  _BYTE *v18; // rdi
  char *v19; // rax
  char v20; // al
  const char *v21; // rax
  char *v22; // r14
  size_t v23; // rax
  __int64 v24; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v25; // [rsp+10h] [rbp-C0h]
  __int64 v26; // [rsp+18h] [rbp-B8h]
  __int64 v29; // [rsp+30h] [rbp-A0h]
  __int64 v30; // [rsp+38h] [rbp-98h]
  unsigned __int64 v31; // [rsp+40h] [rbp-90h] BYREF
  __int64 v32; // [rsp+48h] [rbp-88h]
  __int64 v33; // [rsp+50h] [rbp-80h]
  _BYTE *v34; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-68h]
  _BYTE v36[96]; // [rsp+70h] [rbp-60h] BYREF

  v6 = a1;
  if ( *(_DWORD *)(a2 + 24) == 1 )
  {
    v17 = (unsigned __int64 *)(a2 + 192);
    sub_2240AE0((unsigned __int64 *)(a2 + 192), *(unsigned __int64 **)(a2 + 16));
    *(_DWORD *)(a2 + 224) = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 2488LL))(
                              a1,
                              *(_QWORD *)(a2 + 192),
                              *(_QWORD *)(a2 + 200));
LABEL_18:
    if ( !sub_2241AC0((__int64)v17, "X") )
    {
      v19 = *(char **)(a2 + 232);
      if ( v19 )
      {
        v20 = *v19;
        if ( v20 )
        {
          if ( v20 != 17 )
          {
            if ( v20 == 4 || v20 == 23 )
            {
              sub_2241130(v17, 0, *(_QWORD *)(a2 + 200), "i", 1u);
            }
            else
            {
              v21 = (const char *)(*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6 + 2512LL))(
                                    v6,
                                    *(unsigned __int16 *)(a2 + 240),
                                    0);
              v22 = (char *)v21;
              if ( v21 )
              {
                v23 = strlen(v21);
                sub_2241130(v17, 0, *(_QWORD *)(a2 + 200), v22, v23);
                *(_DWORD *)(a2 + 224) = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6 + 2488LL))(
                                          v6,
                                          *(_QWORD *)(a2 + 192),
                                          *(_QWORD *)(a2 + 200));
              }
            }
          }
        }
      }
    }
    return;
  }
  sub_349C910((__int64)&v34, a1, a2, a4, a5, a6);
  if ( v35 )
  {
    v26 = v35;
    v9 = v34;
    v29 = v35 - 1;
    v10 = 0;
    while ( 1 )
    {
      v11 = &v9[24 * v10];
      v30 = 24 * v10;
      if ( (unsigned int)(*((_DWORD *)v11 + 4) - 4) > 1 )
        break;
      if ( a3 )
      {
        v12 = *(_QWORD *)a1;
        v32 = 0;
        v33 = 0;
        v13 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, unsigned __int64 *, __int64))(v12 + 2520);
        v31 = 0;
        v13(a1, a3, a4, *(_QWORD *)v11, *((_QWORD *)v11 + 1), &v31, a5);
        v14 = v31;
        v15 = v32;
        if ( v31 )
        {
          v24 = v32;
          v25 = v31;
          j_j___libc_free_0(v31);
          v15 = v24;
          v14 = v25;
        }
        v9 = v34;
        if ( v15 != v14 )
        {
          v6 = a1;
          v9 = &v34[24 * v10];
          goto LABEL_12;
        }
      }
      v16 = (unsigned int)(v10 + 1);
      if ( v29 == v10 )
      {
        v30 = 0;
        v6 = a1;
        goto LABEL_12;
      }
      if ( v26 == ++v10 )
      {
        v6 = a1;
        v30 = 24 * v16;
        v9 += 24 * v16;
        goto LABEL_12;
      }
    }
    v6 = a1;
    v9 += 24 * v10;
LABEL_12:
    v17 = (unsigned __int64 *)(a2 + 192);
    sub_2241130((unsigned __int64 *)(a2 + 192), 0, *(_QWORD *)(a2 + 200), *(_BYTE **)v9, *((_QWORD *)v9 + 1));
    v18 = v34;
    *(_DWORD *)(a2 + 224) = *(_DWORD *)&v34[v30 + 16];
    if ( v18 != v36 )
      _libc_free((unsigned __int64)v18);
    goto LABEL_18;
  }
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
