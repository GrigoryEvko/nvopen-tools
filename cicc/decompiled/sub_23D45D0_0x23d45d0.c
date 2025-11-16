// Function: sub_23D45D0
// Address: 0x23d45d0
//
__int64 __fastcall sub_23D45D0(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v3; // rdi
  int v4; // edx
  unsigned int v6; // r15d
  __int64 v7; // rdx
  _BYTE *v8; // rcx
  __int64 v9; // r8
  _BYTE *v10; // rbx
  _BYTE *v11; // rdx
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rbx
  _BYTE *v16; // r13
  _BYTE *v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rax
  __int64 v20; // rsi
  bool v21; // al
  __int64 v22; // [rsp+8h] [rbp-188h]
  const void **v23; // [rsp+8h] [rbp-188h]
  _BYTE *v24; // [rsp+8h] [rbp-188h]
  _BYTE *v25; // [rsp+20h] [rbp-170h] BYREF
  unsigned __int8 *v26; // [rsp+28h] [rbp-168h] BYREF
  _BYTE *v27; // [rsp+30h] [rbp-160h] BYREF
  __int64 v28; // [rsp+38h] [rbp-158h] BYREF
  __int64 v29; // [rsp+40h] [rbp-150h] BYREF
  unsigned int v30; // [rsp+48h] [rbp-148h]
  int v31; // [rsp+4Ch] [rbp-144h]
  unsigned __int64 v32; // [rsp+50h] [rbp-140h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-138h]
  unsigned __int64 v34; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-128h]
  unsigned __int64 v36; // [rsp+70h] [rbp-120h] BYREF
  unsigned int v37; // [rsp+78h] [rbp-118h]
  unsigned __int64 v38; // [rsp+80h] [rbp-110h] BYREF
  unsigned int v39; // [rsp+88h] [rbp-108h]
  unsigned __int64 v40; // [rsp+90h] [rbp-100h] BYREF
  unsigned int v41; // [rsp+98h] [rbp-F8h]
  const void **v42[4]; // [rsp+A0h] [rbp-F0h] BYREF
  __int16 v43; // [rsp+C0h] [rbp-D0h]
  unsigned __int64 v44; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v45; // [rsp+D8h] [rbp-B8h] BYREF
  const void **v46; // [rsp+E0h] [rbp-B0h] BYREF
  const void **v47[21]; // [rsp+E8h] [rbp-A8h] BYREF

  v1 = 0;
  if ( *(_BYTE *)a1 == 55 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(unsigned __int8 *)(v3 + 8);
    if ( (unsigned int)(v4 - 17) <= 1 )
      LOBYTE(v4) = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
    v1 = 0;
    if ( (_BYTE)v4 == 12 )
    {
      v6 = sub_BCB060(v3);
      if ( v6 - 9 <= 0x77 && (v6 & 7) == 0 )
      {
        LODWORD(v45) = 8;
        v44 = 85;
        sub_C47700((__int64)&v32, v6, (__int64)&v44);
        if ( (unsigned int)v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        LODWORD(v45) = 8;
        v44 = 51;
        sub_C47700((__int64)&v34, v6, (__int64)&v44);
        if ( (unsigned int)v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        LODWORD(v45) = 8;
        v44 = 15;
        sub_C47700((__int64)&v36, v6, (__int64)&v44);
        if ( (unsigned int)v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        LODWORD(v45) = 8;
        v44 = 1;
        sub_C47700((__int64)&v38, v6, (__int64)&v44);
        if ( (unsigned int)v45 > 0x40 && v44 )
          j_j___libc_free_0_0(v44);
        v41 = v6;
        if ( v6 > 0x40 )
          sub_C43690((__int64)&v40, v6 - 8, 0);
        else
          v40 = v6 - 8;
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v7 = *(_QWORD *)(a1 - 8);
        else
          v7 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        v8 = *(_BYTE **)v7;
        v9 = *(_QWORD *)(v7 + 32);
        v45 = (__int64)&v38;
        v44 = (unsigned __int64)&v25;
        if ( *v8 != 46 )
          goto LABEL_23;
        v22 = v9;
        if ( !*((_QWORD *)v8 - 8) )
          goto LABEL_23;
        v25 = (_BYTE *)*((_QWORD *)v8 - 8);
        if ( !sub_10080A0((const void ***)&v45, *((_QWORD *)v8 - 4)) )
          goto LABEL_23;
        v42[0] = (const void **)&v40;
        if ( !sub_10080A0(v42, v22) )
          goto LABEL_23;
        v47[0] = (const void **)&v36;
        v10 = v25;
        v44 = (unsigned __int64)&v26;
        v45 = 4;
        v46 = (const void **)&v26;
        if ( *v25 != 57 )
          goto LABEL_23;
        v11 = (_BYTE *)*((_QWORD *)v25 - 8);
        if ( *v11 != 42 )
          goto LABEL_23;
        v12 = (_BYTE *)*((_QWORD *)v11 - 8);
        if ( *v12 == 55
          && *((_QWORD *)v12 - 8)
          && (v26 = (unsigned __int8 *)*((_QWORD *)v12 - 8),
              v24 = v11,
              v21 = sub_F17ED0(&v45, *((_QWORD *)v12 - 4)),
              v11 = v24,
              v21) )
        {
          v13 = (_BYTE *)*((_QWORD *)v24 - 4);
          if ( v13 == *v46 )
          {
LABEL_53:
            if ( sub_10080A0(v47, *((_QWORD *)v10 - 4)) )
            {
              v47[0] = (const void **)2;
              v44 = (unsigned __int64)&v27;
              v45 = (__int64)&v34;
              v46 = (const void **)&v27;
              v47[1] = (const void **)&v34;
              if ( sub_23D1310((__int64)&v44, 13, v26) && *v27 == 44 )
              {
                v15 = *((_QWORD *)v27 - 8);
                if ( v15 )
                {
                  v16 = (_BYTE *)*((_QWORD *)v27 - 4);
                  if ( v16 )
                  {
                    v44 = *((_QWORD *)v27 - 8);
                    v45 = 1;
                    v46 = (const void **)&v32;
                    if ( *v16 == 57 )
                    {
                      v17 = (_BYTE *)*((_QWORD *)v16 - 8);
                      if ( *v17 == 55 && v15 == *((_QWORD *)v17 - 8) && sub_F17ED0(&v45, *((_QWORD *)v17 - 4)) )
                      {
                        LOBYTE(v18) = sub_10080A0(&v46, *((_QWORD *)v16 - 4));
                        v1 = v18;
                        if ( (_BYTE)v18 )
                        {
                          sub_23D0AB0((__int64)&v44, a1, 0, 0, 0);
                          v19 = *(_QWORD *)(a1 + 8);
                          v31 = 0;
                          v43 = 257;
                          v29 = v15;
                          v28 = v19;
                          v20 = sub_B33D10((__int64)&v44, 0x42u, (__int64)&v28, 1, (int)&v29, 1, v30, (__int64)v42);
                          sub_BD84D0(a1, v20);
                          sub_F94A20(&v44, v20);
LABEL_24:
                          if ( v41 > 0x40 && v40 )
                            j_j___libc_free_0_0(v40);
                          if ( v39 > 0x40 && v38 )
                            j_j___libc_free_0_0(v38);
                          if ( v37 > 0x40 && v36 )
                            j_j___libc_free_0_0(v36);
                          if ( v35 > 0x40 && v34 )
                            j_j___libc_free_0_0(v34);
                          if ( v33 > 0x40 && v32 )
                            j_j___libc_free_0_0(v32);
                          return v1;
                        }
                      }
                    }
                  }
                }
              }
            }
LABEL_23:
            v1 = 0;
            goto LABEL_24;
          }
        }
        else
        {
          v13 = (_BYTE *)*((_QWORD *)v11 - 4);
        }
        if ( *v13 != 55 )
          goto LABEL_23;
        v14 = *((_QWORD *)v13 - 8);
        v23 = (const void **)v11;
        if ( !v14 )
          goto LABEL_23;
        *(_QWORD *)v44 = v14;
        if ( !sub_F17ED0(&v45, *((_QWORD *)v13 - 4)) || *(v23 - 8) != *v46 )
          goto LABEL_23;
        goto LABEL_53;
      }
    }
  }
  return v1;
}
