// Function: sub_37EB4A0
// Address: 0x37eb4a0
//
__int64 __fastcall sub_37EB4A0(__int64 *a1)
{
  __int64 *v1; // r14
  __int64 v2; // r12
  __int64 *v3; // r13
  int v4; // eax
  char v5; // al
  __int64 *v6; // rbx
  unsigned int v7; // r12d
  _BYTE *v8; // rdi
  __int64 *v9; // r13
  unsigned int v10; // r14d
  unsigned __int8 *v11; // rax
  size_t v12; // rdx
  unsigned int v13; // r15d
  __int64 v14; // rdi
  const char *v15; // r14
  _BYTE *v16; // rsi
  __int64 v17; // rax
  _WORD *v18; // rdx
  __int64 v19; // r15
  __int64 v20; // rbx
  _BYTE *v21; // rdi
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  size_t v27; // [rsp+18h] [rbp-188h]
  __int64 *v29; // [rsp+28h] [rbp-178h]
  __int64 v30; // [rsp+30h] [rbp-170h]
  __int64 *v31; // [rsp+38h] [rbp-168h]
  const char *v32; // [rsp+40h] [rbp-160h] BYREF
  __int64 v33; // [rsp+48h] [rbp-158h]
  __int64 v34; // [rsp+50h] [rbp-150h]
  __int64 v35; // [rsp+58h] [rbp-148h]
  void *dest; // [rsp+60h] [rbp-140h]
  __int64 v37; // [rsp+68h] [rbp-138h]
  const char **v38; // [rsp+70h] [rbp-130h]
  _BYTE *v39; // [rsp+80h] [rbp-120h] BYREF
  __int64 v40; // [rsp+88h] [rbp-118h]
  _BYTE v41[64]; // [rsp+90h] [rbp-110h] BYREF
  const char *v42; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v43; // [rsp+D8h] [rbp-C8h]
  __int64 v44; // [rsp+E0h] [rbp-C0h]
  _BYTE v45[184]; // [rsp+E8h] [rbp-B8h] BYREF

  v1 = (__int64 *)a1[41];
  v39 = v41;
  v40 = 0x800000000LL;
  v31 = a1 + 40;
  if ( v1 == a1 + 40 )
  {
    return 0;
  }
  else
  {
    do
    {
      v2 = v1[7];
      v3 = v1 + 6;
      if ( (__int64 *)v2 != v1 + 6 )
      {
        do
        {
          while ( 1 )
          {
            v4 = *(_DWORD *)(v2 + 44);
            if ( (v4 & 4) != 0 || (v4 & 8) == 0 )
              v5 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v2 + 16) + 24LL) >> 7;
            else
              v5 = sub_2E88A90(v2, 128, 1);
            if ( v5 && (*(_DWORD *)(v2 + 40) & 0xFFFFFF) != 0 )
            {
              v19 = *(_QWORD *)(v2 + 32) + 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
              v20 = *(_QWORD *)(v2 + 32);
              while ( 1 )
              {
                if ( *(_BYTE *)v20 == 10 )
                {
                  v21 = *(_BYTE **)(v20 + 24);
                  if ( !*v21 )
                  {
                    if ( (unsigned __int8)sub_B2D610((__int64)v21, 53) )
                      break;
                  }
                }
                v20 += 40;
                if ( v19 == v20 )
                  goto LABEL_10;
              }
              v24 = (unsigned int)v40;
              v25 = (unsigned int)v40 + 1LL;
              if ( v25 > HIDWORD(v40) )
              {
                sub_C8D5F0((__int64)&v39, v41, v25, 8u, v22, v23);
                v24 = (unsigned int)v40;
              }
              *(_QWORD *)&v39[8 * v24] = v2;
              LODWORD(v40) = v40 + 1;
            }
LABEL_10:
            if ( (*(_BYTE *)v2 & 4) == 0 )
              break;
            v2 = *(_QWORD *)(v2 + 8);
            if ( v3 == (__int64 *)v2 )
              goto LABEL_12;
          }
          while ( (*(_BYTE *)(v2 + 44) & 8) != 0 )
            v2 = *(_QWORD *)(v2 + 8);
          v2 = *(_QWORD *)(v2 + 8);
        }
        while ( v3 != (__int64 *)v2 );
      }
LABEL_12:
      v1 = (__int64 *)v1[1];
    }
    while ( v31 != v1 );
    v6 = a1;
    v7 = 0;
    v8 = v39;
    if ( (_DWORD)v40 )
    {
      v9 = (__int64 *)v39;
      v10 = 0;
      v29 = (__int64 *)&v39[8 * (unsigned int)v40];
      while ( 1 )
      {
        v17 = *v9;
        v43 = 0;
        v30 = v17;
        v44 = 128;
        v42 = v45;
        v37 = 0x100000000LL;
        v33 = 2;
        v34 = 0;
        v32 = (const char *)&unk_49DD288;
        v35 = 0;
        v38 = &v42;
        dest = 0;
        sub_CB5980((__int64)&v32, 0, 0, 0);
        v18 = dest;
        if ( (unsigned __int64)(v35 - (_QWORD)dest) > 6 )
        {
          *(_DWORD *)dest = 1734763300;
          v18[2] = 27251;
          *((_BYTE *)v18 + 6) = 95;
          dest = (char *)dest + 7;
        }
        else
        {
          sub_CB6200((__int64)&v32, "$cfgsj_", 7u);
        }
        v11 = (unsigned __int8 *)sub_2E791E0(v6);
        if ( v12 > v35 - (__int64)dest )
        {
          sub_CB6200((__int64)&v32, v11, v12);
        }
        else if ( v12 )
        {
          v27 = v12;
          memcpy(dest, v11, v12);
          dest = (char *)dest + v27;
        }
        v13 = v10 + 1;
        sub_CB59D0((__int64)&v32, v10);
        v32 = (const char *)&unk_49DD388;
        sub_CB5840((__int64)&v32);
        v14 = v6[3];
        LOWORD(dest) = 261;
        v32 = v42;
        v33 = v43;
        v15 = (const char *)sub_E6C460(v14, &v32);
        sub_2E87EC0(v30, (__int64)v6, (__int64)v15);
        v32 = v15;
        v16 = (_BYTE *)v6[49];
        if ( v16 == (_BYTE *)v6[50] )
        {
          sub_31DFB90((__int64)(v6 + 48), v16, &v32);
        }
        else
        {
          if ( v16 )
          {
            *(_QWORD *)v16 = v15;
            v16 = (_BYTE *)v6[49];
          }
          v6[49] = (__int64)(v16 + 8);
        }
        if ( v42 != v45 )
          _libc_free((unsigned __int64)v42);
        if ( v29 == ++v9 )
          break;
        v10 = v13;
      }
      v8 = v39;
      v7 = 1;
    }
    if ( v8 != v41 )
      _libc_free((unsigned __int64)v8);
  }
  return v7;
}
