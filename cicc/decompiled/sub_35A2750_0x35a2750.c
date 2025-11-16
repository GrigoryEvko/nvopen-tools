// Function: sub_35A2750
// Address: 0x35a2750
//
unsigned __int64 __fastcall sub_35A2750(__int64 a1, __int64 a2, int a3)
{
  _QWORD *i; // r14
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rsi
  unsigned int v8; // r8d
  _QWORD *v9; // rax
  _QWORD *v10; // rcx
  __int64 v11; // rsi
  int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r13
  unsigned int v20; // eax
  __int64 v21; // r9
  _BYTE *v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // rax
  int v25; // edx
  __int64 *v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // rax
  __int64 *v29; // rdi
  __int64 *v30; // r12
  __int64 *v31; // r13
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // rax
  unsigned __int64 v37; // r8
  __int64 *v38; // rax
  int v39; // eax
  int v40; // r9d
  _QWORD *v41; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v42; // [rsp+0h] [rbp-C0h]
  unsigned __int64 v43; // [rsp+10h] [rbp-B0h]
  __int64 j; // [rsp+28h] [rbp-98h]
  unsigned __int64 v47; // [rsp+30h] [rbp-90h]
  __int64 *v48; // [rsp+40h] [rbp-80h] BYREF
  __int64 v49; // [rsp+48h] [rbp-78h]
  _BYTE v50[112]; // [rsp+50h] [rbp-70h] BYREF

  for ( i = (_QWORD *)sub_2E318E0(a2); ; i = (_QWORD *)v47 )
  {
    result = *(_QWORD *)sub_2E311E0(a2) & 0xFFFFFFFFFFFFFFF8LL;
    if ( i == (_QWORD *)result )
      break;
    v6 = *(unsigned int *)(a1 + 280);
    v7 = *(_QWORD *)(a1 + 264);
    v47 = *i & 0xFFFFFFFFFFFFFFF8LL;
    if ( !(_DWORD)v6 )
      goto LABEL_44;
    v8 = (v6 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
    v9 = (_QWORD *)(v7 + 16LL * v8);
    v10 = (_QWORD *)*v9;
    if ( i != (_QWORD *)*v9 )
    {
      v39 = 1;
      while ( v10 != (_QWORD *)-4096LL )
      {
        v40 = v39 + 1;
        v8 = (v6 - 1) & (v39 + v8);
        v9 = (_QWORD *)(v7 + 16LL * v8);
        v10 = (_QWORD *)*v9;
        if ( i == (_QWORD *)*v9 )
          goto LABEL_6;
        v39 = v40;
      }
LABEL_44:
      v11 = (__int64)i;
      goto LABEL_8;
    }
LABEL_6:
    if ( v9 == (_QWORD *)(v7 + 16 * v6) )
      goto LABEL_44;
    v11 = v9[1];
LABEL_8:
    v12 = sub_3598DB0(*(_QWORD *)a1, v11);
    if ( v12 != -1 && a3 > v12 )
    {
      v13 = i[4];
      v14 = v13 + 40LL * (unsigned int)sub_2E88FE0((__int64)i);
      v15 = i[4];
      for ( j = v14; v15 != j; v15 += 40 )
      {
        v16 = *(_QWORD *)(a1 + 24);
        v48 = (__int64 *)v50;
        v49 = 0x400000000LL;
        v17 = *(unsigned int *)(v15 + 8);
        if ( (int)v17 < 0 )
          v18 = *(_QWORD *)(*(_QWORD *)(v16 + 56) + 16 * (v17 & 0x7FFFFFFF) + 8);
        else
          v18 = *(_QWORD *)(*(_QWORD *)(v16 + 304) + 8 * v17);
        if ( v18 )
        {
          if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 )
          {
            while ( 1 )
            {
              v18 = *(_QWORD *)(v18 + 32);
              if ( !v18 )
                break;
              if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
                goto LABEL_15;
            }
          }
          else
          {
LABEL_15:
            v19 = *(_QWORD *)(v18 + 16);
LABEL_16:
            v20 = sub_35A2540(a1, *(_DWORD *)(*(_QWORD *)(v19 + 32) + 8LL), i[3]);
            v22 = (_BYTE *)HIDWORD(v49);
            v23 = v20;
            v24 = (unsigned int)v49;
            v25 = v49;
            if ( (unsigned int)v49 >= (unsigned __int64)HIDWORD(v49) )
            {
              v27 = (unsigned int)v49 + 1LL;
              v37 = v23 | v43 & 0xFFFFFFFF00000000LL;
              v43 = v37;
              if ( HIDWORD(v49) < v27 )
              {
                v22 = v50;
                v42 = v37;
                sub_C8D5F0((__int64)&v48, v50, v27, 0x10u, v37, v21);
                v24 = (unsigned int)v49;
                v37 = v42;
              }
              v38 = &v48[2 * v24];
              *v38 = v19;
              v38[1] = v37;
              LODWORD(v49) = v49 + 1;
            }
            else
            {
              v26 = &v48[2 * (unsigned int)v49];
              if ( v26 )
              {
                *v26 = v19;
                *((_DWORD *)v26 + 2) = v23;
                v25 = v49;
              }
              v27 = (unsigned int)(v25 + 1);
              LODWORD(v49) = v27;
            }
            v28 = *(_QWORD *)(v18 + 16);
            while ( 1 )
            {
              v18 = *(_QWORD *)(v18 + 32);
              if ( !v18 )
                break;
              if ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
              {
                v19 = *(_QWORD *)(v18 + 16);
                if ( v28 != v19 )
                  goto LABEL_16;
              }
            }
            v29 = v48;
            v30 = &v48[2 * (unsigned int)v49];
            if ( v30 != v48 )
            {
              v41 = i;
              v31 = v48;
              v32 = a1;
              do
              {
                v33 = *v31;
                v31 += 2;
                v34 = *(_QWORD *)(**(_QWORD **)(v32 + 24) + 16LL);
                v35 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int64, __int64))(*(_QWORD *)v34 + 200LL))(
                                  v34,
                                  v22,
                                  v27,
                                  v23);
                v22 = (_BYTE *)*(unsigned int *)(v15 + 8);
                sub_2E8A790(v33, (int)v22, *((_DWORD *)v31 - 2), 0, v35);
              }
              while ( v30 != v31 );
              a1 = v32;
              v29 = v48;
              i = v41;
            }
            if ( v29 != (__int64 *)v50 )
              _libc_free((unsigned __int64)v29);
          }
        }
      }
      v36 = *(_QWORD *)(a1 + 40);
      if ( v36 )
        sub_2FAD510(*(_QWORD *)(v36 + 32), (__int64)i);
      sub_2E88E20((__int64)i);
    }
  }
  return result;
}
