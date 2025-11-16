// Function: sub_F8A680
// Address: 0xf8a680
//
__int64 __fastcall sub_F8A680(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rdi
  __int64 v4; // rbx
  __int64 i; // r15
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r15
  __int64 v14; // rsi
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // r13
  __int64 *v18; // rbx
  __int64 v19; // r14
  __int64 *v20; // r15
  __int64 *v21; // rax
  _BYTE *v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdi
  _BYTE *v25; // rcx
  _BYTE *v26; // rax
  __int64 v28; // r13
  __int64 *v29; // r14
  __int64 v30; // rbx
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 *v33; // r13
  _BYTE *v34; // rsi
  _BYTE *v35; // r14
  _BYTE *v36; // r13
  _BYTE *v37; // rax
  _BYTE *v38; // [rsp+10h] [rbp-120h]
  __int16 v39; // [rsp+10h] [rbp-120h]
  __int64 v41; // [rsp+30h] [rbp-100h] BYREF
  __int64 v42; // [rsp+38h] [rbp-F8h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-E8h]
  _BYTE v45[32]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 *v46; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+78h] [rbp-B8h]
  _BYTE v48[176]; // [rsp+80h] [rbp-B0h] BYREF

  v3 = (__int64 *)*a1;
  v41 = 0;
  v42 = 0;
  if ( (unsigned __int8)sub_DCFD50(v3, a2, &v41, &v42) )
  {
    v36 = (_BYTE *)sub_F894B0((__int64)a1, v41);
    v37 = (_BYTE *)sub_F894B0((__int64)a1, v42);
    return sub_F810E0(a1, 0x16u, v36, v37, 0, 0);
  }
  else
  {
    v46 = (__int64 *)v48;
    v47 = 0x800000000LL;
    v4 = *(_QWORD *)(a2 + 32);
    for ( i = v4 + 8LL * *(_QWORD *)(a2 + 40); v4 != i; LODWORD(v47) = v47 + 1 )
    {
      v6 = *(_QWORD *)(i - 8);
      v9 = sub_F83200((__int64)a1, v6);
      v10 = (unsigned int)v47;
      v11 = (unsigned int)v47 + 1LL;
      if ( v11 > HIDWORD(v47) )
      {
        sub_C8D5F0((__int64)&v46, v48, v11, 0x10u, v7, v8);
        v10 = (unsigned int)v47;
      }
      i -= 8;
      v12 = &v46[2 * v10];
      *v12 = v9;
      v12[1] = v6;
    }
    v13 = 0;
    v14 = *(_QWORD *)(*a1 + 40);
    sub_F86830((__int64)&v46, v14);
    v17 = v46;
    v18 = &v46[2 * (unsigned int)v47];
    if ( v18 != v46 )
    {
      v19 = 0;
      v20 = v46;
      while ( 1 )
      {
        v23 = v20[1];
        if ( !v19 )
          break;
        if ( *(_BYTE *)(*(_QWORD *)(v19 + 8) + 8LL) == 14 )
        {
          v28 = *v20;
          v38 = (_BYTE *)v19;
          v29 = v18;
          v43 = v45;
          v30 = v28;
          v44 = 0x400000000LL;
          do
          {
            if ( *v20 != v30 )
              break;
            v33 = (__int64 *)v20[1];
            if ( *((_WORD *)v33 + 12) == 15 )
            {
              v34 = (_BYTE *)*(v33 - 1);
              if ( *v34 <= 0x1Cu )
                v33 = sub_DD8400(*a1, (__int64)v34);
            }
            v31 = (unsigned int)v44;
            v32 = (unsigned int)v44 + 1LL;
            if ( v32 > HIDWORD(v44) )
            {
              sub_C8D5F0((__int64)&v43, v45, v32, 8u, v15, v16);
              v31 = (unsigned int)v44;
            }
            v20 += 2;
            *(_QWORD *)&v43[8 * v31] = v33;
            LODWORD(v44) = v44 + 1;
          }
          while ( v20 != v29 );
          v18 = v29;
          v35 = v38;
          v39 = *(_WORD *)(a2 + 28) & 7;
          v14 = (__int64)sub_DC7EB0((__int64 *)*a1, (__int64)&v43, 0, 0);
          v19 = (__int64)sub_F8A290((__int64)a1, v14, v35, v39);
          if ( v43 != v45 )
            _libc_free(v43, v14);
          goto LABEL_9;
        }
        v24 = v20[1];
        v20 += 2;
        if ( sub_D969D0(v24) )
        {
          v21 = sub_DCAF50((__int64 *)*a1, v23, 0);
          v22 = (_BYTE *)sub_F894B0((__int64)a1, (__int64)v21);
          v14 = 15;
          v19 = sub_F810E0(a1, 0xFu, (_BYTE *)v19, v22, 0, 1);
LABEL_9:
          if ( v18 == v20 )
            goto LABEL_16;
        }
        else
        {
          v25 = (_BYTE *)sub_F894B0((__int64)a1, v23);
          if ( *(_BYTE *)v19 <= 0x15u )
          {
            v26 = (_BYTE *)v19;
            v19 = (__int64)v25;
            v25 = v26;
          }
          v14 = 13;
          v19 = sub_F810E0(a1, 0xDu, (_BYTE *)v19, v25, *(_BYTE *)(a2 + 28) & 7, 1);
          if ( v18 == v20 )
          {
LABEL_16:
            v17 = v46;
            v13 = v19;
            goto LABEL_17;
          }
        }
      }
      v14 = v20[1];
      v20 += 2;
      v19 = sub_F894B0((__int64)a1, v14);
      goto LABEL_9;
    }
LABEL_17:
    if ( v17 != (__int64 *)v48 )
      _libc_free(v17, v14);
  }
  return v13;
}
