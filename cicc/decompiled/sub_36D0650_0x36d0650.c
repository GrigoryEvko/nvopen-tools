// Function: sub_36D0650
// Address: 0x36d0650
//
__int64 __fastcall sub_36D0650(__int64 a1, __int64 a2)
{
  const char *v2; // r14
  __int64 v3; // r13
  const char *v4; // rax
  const char *v5; // rbx
  __int64 v6; // rdx
  const char *v7; // r15
  unsigned __int64 *v8; // rdi
  const char *v9; // r14
  const char *v10; // r15
  int v11; // r12d
  unsigned __int64 v12; // rdx
  size_t v13; // rbx
  unsigned __int64 v14; // rcx
  size_t v15; // rax
  const char *v16; // r14
  __int64 v17; // r13
  const char *v18; // rax
  const char *v19; // rbx
  __int64 v20; // rdx
  const char *v21; // r15
  unsigned __int64 *v22; // rdi
  const char *v23; // r14
  const char *v24; // r15
  int v25; // r12d
  unsigned __int64 v26; // rdx
  size_t v27; // rbx
  unsigned __int64 v28; // rcx
  size_t v29; // rax
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v34; // [rsp+20h] [rbp-D0h]
  size_t v35; // [rsp+30h] [rbp-C0h]
  size_t v36; // [rsp+30h] [rbp-C0h]
  _BYTE v37[3]; // [rsp+4Dh] [rbp-A3h] BYREF
  _QWORD *v38; // [rsp+50h] [rbp-A0h] BYREF
  size_t v39; // [rsp+58h] [rbp-98h]
  _QWORD v40[2]; // [rsp+60h] [rbp-90h] BYREF
  _QWORD *v41; // [rsp+70h] [rbp-80h] BYREF
  size_t v42; // [rsp+78h] [rbp-78h]
  _QWORD v43[2]; // [rsp+80h] [rbp-70h] BYREF
  const char *v44[4]; // [rsp+90h] [rbp-60h] BYREF
  __int16 v45; // [rsp+B0h] [rbp-40h]

  v2 = (const char *)&v38;
  v3 = *(_QWORD *)(a2 + 16);
  v34 = a2 + 8;
  if ( v3 != a2 + 8 )
  {
    while ( v3 )
    {
      if ( (*(_BYTE *)(v3 - 24) & 0xFu) - 7 <= 1 )
      {
        v4 = sub_BD5D20(v3 - 56);
        v38 = v40;
        v5 = v4;
        v39 = 0;
        LOBYTE(v40[0]) = 0;
        v7 = &v4[v6];
        sub_2240E30((__int64)v2, v6 + 4);
        if ( v5 != v7 )
        {
          v8 = (unsigned __int64 *)v2;
          v9 = v7;
          v10 = v5;
          do
          {
            while ( 1 )
            {
              v11 = *(unsigned __int8 *)v10;
              if ( (unsigned __int8)((*v10 & 0xDF) - 65) > 0x19u )
              {
                if ( (unsigned __int8)(v11 - 36) > 0x3Bu )
                  break;
                v31 = 0x8000000003FF001LL;
                if ( !_bittest64(&v31, (unsigned int)(v11 - 36)) )
                  break;
              }
              v12 = (unsigned __int64)v38;
              v13 = v39;
              v14 = 15;
              if ( v38 != v40 )
                v14 = v40[0];
              v15 = v39 + 1;
              if ( v39 + 1 > v14 )
              {
                v35 = v39 + 1;
                sub_2240BB0(v8, v39, 0, 0, 1u);
                v12 = (unsigned __int64)v38;
                v15 = v35;
              }
              *(_BYTE *)(v12 + v13) = v11;
              ++v10;
              v39 = v15;
              *((_BYTE *)v38 + v13 + 1) = 0;
              if ( v9 == v10 )
                goto LABEL_15;
            }
            qmemcpy(v37, "_$_", sizeof(v37));
            if ( 0x3FFFFFFFFFFFFFFFLL - v39 <= 2 )
LABEL_42:
              sub_4262D8((__int64)"basic_string::append");
            ++v10;
            sub_2241490(v8, v37, 3u);
          }
          while ( v9 != v10 );
LABEL_15:
          v2 = (const char *)v8;
        }
        v44[0] = v2;
        v45 = 260;
        sub_BD6B50((unsigned __int8 *)(v3 - 56), v44);
        if ( v38 != v40 )
          j_j___libc_free_0((unsigned __int64)v38);
      }
      v3 = *(_QWORD *)(v3 + 8);
      if ( v34 == v3 )
        goto LABEL_19;
    }
LABEL_43:
    BUG();
  }
LABEL_19:
  v16 = (const char *)&v41;
  v17 = *(_QWORD *)(a2 + 32);
  if ( a2 + 24 != v17 )
  {
    while ( v17 )
    {
      if ( (*(_BYTE *)(v17 - 24) & 0xFu) - 7 <= 1 )
      {
        v18 = sub_BD5D20(v17 - 56);
        v41 = v43;
        v19 = v18;
        LOBYTE(v43[0]) = 0;
        v42 = 0;
        v21 = &v18[v20];
        sub_2240E30((__int64)v16, v20 + 4);
        if ( v21 != v19 )
        {
          v22 = (unsigned __int64 *)v16;
          v23 = v21;
          v24 = v19;
          do
          {
            while ( 1 )
            {
              v25 = *(unsigned __int8 *)v24;
              if ( (unsigned __int8)((*v24 & 0xDF) - 65) > 0x19u )
              {
                if ( (unsigned __int8)(v25 - 36) > 0x3Bu )
                  break;
                v32 = 0x8000000003FF001LL;
                if ( !_bittest64(&v32, (unsigned int)(v25 - 36)) )
                  break;
              }
              v26 = (unsigned __int64)v41;
              v27 = v42;
              v28 = 15;
              if ( v41 != v43 )
                v28 = v43[0];
              v29 = v42 + 1;
              if ( v42 + 1 > v28 )
              {
                v36 = v42 + 1;
                sub_2240BB0(v22, v42, 0, 0, 1u);
                v26 = (unsigned __int64)v41;
                v29 = v36;
              }
              *(_BYTE *)(v26 + v27) = v25;
              ++v24;
              v42 = v29;
              *((_BYTE *)v41 + v27 + 1) = 0;
              if ( v23 == v24 )
                goto LABEL_33;
            }
            qmemcpy(v44, "_$_", 3);
            if ( 0x3FFFFFFFFFFFFFFFLL - v42 <= 2 )
              goto LABEL_42;
            ++v24;
            sub_2241490(v22, (char *)v44, 3u);
          }
          while ( v23 != v24 );
LABEL_33:
          v16 = (const char *)v22;
        }
        v44[0] = v16;
        v45 = 260;
        sub_BD6B50((unsigned __int8 *)(v17 - 56), v44);
        if ( v41 != v43 )
          j_j___libc_free_0((unsigned __int64)v41);
      }
      v17 = *(_QWORD *)(v17 + 8);
      if ( a2 + 24 == v17 )
        return 1;
    }
    goto LABEL_43;
  }
  return 1;
}
