// Function: sub_2A04840
// Address: 0x2a04840
//
__int16 __fastcall sub_2A04840(_QWORD *a1, unsigned __int8 **a2, unsigned int *a3)
{
  unsigned __int64 v3; // r15
  unsigned __int8 *v4; // r8
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 *v10; // r12
  __int64 *v11; // r13
  __int64 *v12; // r14
  __int64 v13; // rax
  _QWORD *v14; // rax
  _BYTE *v15; // r14
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // r12
  unsigned int v21; // ebx
  __int64 *v22; // rdi
  __int64 v23; // r12
  unsigned int v24; // eax
  __int64 *v25; // rdi
  _BYTE *v26; // r12
  unsigned int v27; // r14d
  __int64 v28; // r13
  __int64 v29; // rcx
  int v30; // r12d
  __int64 v31; // rax
  __int64 v32; // r15
  unsigned int v33; // eax
  _QWORD *v35; // [rsp-A8h] [rbp-A8h]
  __int64 v36; // [rsp-90h] [rbp-90h]
  __int64 v37; // [rsp-88h] [rbp-88h]
  unsigned __int64 v38; // [rsp-88h] [rbp-88h]
  unsigned int v39; // [rsp-7Ch] [rbp-7Ch]
  unsigned int v40; // [rsp-78h] [rbp-78h]
  unsigned __int64 v41; // [rsp-70h] [rbp-70h]
  int v42; // [rsp-68h] [rbp-68h] BYREF
  int v43; // [rsp-64h] [rbp-64h] BYREF
  __int64 v44; // [rsp-60h] [rbp-60h] BYREF
  _QWORD *v45; // [rsp-58h] [rbp-58h] BYREF
  __int64 v46; // [rsp-50h] [rbp-50h]
  _BYTE *v47; // [rsp-48h] [rbp-48h] BYREF
  __int64 v48; // [rsp-40h] [rbp-40h]
  unsigned __int64 v49; // [rsp-10h] [rbp-10h]

  v4 = *a2;
  v5 = *a3;
  v6 = *((_QWORD *)*a2 + 1);
  if ( *(_BYTE *)(v6 + 8) == 12 && (unsigned int)v5 <= 3 )
  {
    v49 = v3;
    LOWORD(v6) = *v4;
    v7 = (_QWORD *)*a1;
    if ( (_BYTE)v6 == 57 || (_BYTE)v6 == 58 )
    {
      v6 = *((_QWORD *)v4 - 8);
      if ( v6 )
      {
        v28 = *((_QWORD *)v4 - 4);
        if ( v28 )
        {
          v29 = *v7;
          v30 = v5 + 1;
          v45 = (_QWORD *)*((_QWORD *)v4 - 8);
          v43 = v5 + 1;
          if ( !*(_QWORD *)(v29 + 16)
            || (a2 = (unsigned __int8 **)&v45,
                a1 = (_QWORD *)v29,
                (*(void (__fastcall **)(__int64, _QWORD **, int *))(v29 + 24))(v29, &v45, &v43),
                v31 = *v7,
                v44 = v28,
                v42 = v30,
                !*(_QWORD *)(v31 + 16)) )
          {
            sub_4263D6(a1, a2, v5);
          }
          LOWORD(v6) = (*(__int64 (__fastcall **)(__int64, __int64 *, int *))(v31 + 24))(v31, &v44, &v42);
        }
      }
    }
    else if ( (_BYTE)v6 == 82 )
    {
      v8 = *((_QWORD *)v4 - 8);
      if ( v8 )
      {
        v9 = *((_QWORD *)v4 - 4);
        if ( v9 )
        {
          v41 = sub_B53900((__int64)v4);
          v37 = v41 & 0xFFFFFFFFFFLL;
          v39 = v41;
          v10 = sub_DD8400(v7[1], v8);
          v11 = sub_DD8400(v7[1], v9);
          LOWORD(v6) = (unsigned __int16)sub_DC3AF0(v7[1], v41 & 0xFFFFFFFFFFLL, v10, v11) >> 8;
          if ( !(_WORD)v6 )
          {
            if ( *((_WORD *)v10 + 12) != 8 )
            {
              if ( *((_WORD *)v11 + 12) != 8 )
                return v6;
              v39 = sub_B52F50(v41);
              v37 = (unsigned int)v41;
              v6 = (__int64)v10;
              v10 = v11;
              v11 = (__int64 *)v6;
            }
            if ( v10[5] == 2 )
            {
              v6 = v7[2];
              if ( v10[6] == v6 )
              {
                if ( v39 - 32 <= 1 && (*((_BYTE *)v10 + 28) & 1) != 0
                  || (v6 = sub_DC1950(v7[1], (__int64)v10, v39), v45 = (_QWORD *)v6, BYTE4(v6)) )
                {
                  v12 = (__int64 *)v7[1];
                  v40 = *(_DWORD *)v7[3];
                  v13 = sub_D95540((__int64)v10);
                  v14 = sub_DA2C50((__int64)v12, v13, v40, 0);
                  v15 = sub_DD0540((__int64)v10, (__int64)v14, v12);
                  v38 = v39 | v37 & 0xFFFFFFFF00000000LL;
                  if ( !(unsigned __int8)sub_DC3A60(v7[1], v38, v15, v11) )
                  {
                    v39 = sub_B52870(v39);
                    v38 &= 0xFFFFFF00FFFFFFFFLL;
                  }
                  v19 = sub_D33D80(v10, v7[1], v16, v17, v18);
                  v20 = v7[4];
                  v36 = v19;
                  if ( v40 < **(_DWORD **)v20 )
                  {
                    v35 = v7;
                    v21 = v40;
                    do
                    {
                      v3 = v39 | v3 & 0xFFFFFF0000000000LL;
                      if ( !(unsigned __int8)sub_DC3A60(*(_QWORD *)(v20 + 8), v3, v15, v11) )
                        break;
                      v22 = *(__int64 **)(v20 + 8);
                      v47 = v15;
                      v48 = v36;
                      v45 = &v47;
                      v46 = 0x200000002LL;
                      v15 = sub_DC7EB0(v22, (__int64)&v45, 0, 0);
                      if ( v45 != &v47 )
                        _libc_free((unsigned __int64)v45);
                      ++v21;
                    }
                    while ( **(_DWORD **)v20 > v21 );
                    v40 = v21;
                    v7 = v35;
                  }
                  v23 = *(_QWORD *)(v20 + 8);
                  v24 = sub_B52870(v39);
                  LOWORD(v6) = sub_DC3A60(v23, v24, v15, v11);
                  if ( (_BYTE)v6 )
                  {
                    v25 = (__int64 *)v7[1];
                    v47 = v15;
                    v48 = v36;
                    v45 = &v47;
                    v46 = 0x200000002LL;
                    v26 = sub_DC7EB0(v25, (__int64)&v45, 0, 0);
                    if ( v45 != &v47 )
                      _libc_free((unsigned __int64)v45);
                    if ( v39 - 32 <= 1 )
                    {
                      v32 = v7[1];
                      v33 = sub_B52870(v39);
                      if ( !(unsigned __int8)sub_DC3A60(v32, v33, v26, v11)
                        && !(unsigned __int8)sub_DC3A60(v7[1], v39 | v38 & 0xFFFFFFFF00000000LL, v15, v11)
                        && (unsigned __int8)sub_DC3A60(v7[1], v39 | v38 & 0xFFFFFFFF00000000LL, v26, v11) )
                      {
                        v6 = v7[5];
                        if ( *(_DWORD *)v6 <= v40 )
                          return v6;
                        ++v40;
                      }
                    }
                    v6 = v7[3];
                    v27 = v40;
                    if ( *(_DWORD *)v6 >= v40 )
                      v27 = *(_DWORD *)v6;
                    *(_DWORD *)v6 = v27;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return v6;
}
