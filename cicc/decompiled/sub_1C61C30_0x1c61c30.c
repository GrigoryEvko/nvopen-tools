// Function: sub_1C61C30
// Address: 0x1c61c30
//
__int64 __fastcall sub_1C61C30(_QWORD *a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  _QWORD *v5; // rax
  _QWORD *v7; // r15
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // r12
  unsigned __int64 v11; // rax
  char v12; // r11
  char v13; // al
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  _BYTE *v24; // rax
  __int64 v25; // [rsp+8h] [rbp-128h]
  __int64 v26; // [rsp+10h] [rbp-120h]
  unsigned __int8 v27; // [rsp+28h] [rbp-108h]
  __int64 v28[2]; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v29; // [rsp+48h] [rbp-E8h] BYREF
  __m128i v30; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+60h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+68h] [rbp-C8h]
  __int64 v33; // [rsp+70h] [rbp-C0h]
  __int64 v34; // [rsp+78h] [rbp-B8h]
  const __m128i *v35[4]; // [rsp+80h] [rbp-B0h] BYREF
  char v36; // [rsp+A0h] [rbp-90h]
  __int64 v37[2]; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD *v38; // [rsp+C0h] [rbp-70h]
  __int64 v39; // [rsp+C8h] [rbp-68h]
  __int64 v40; // [rsp+D0h] [rbp-60h]
  __int64 v41; // [rsp+D8h] [rbp-58h]
  _QWORD *v42; // [rsp+E0h] [rbp-50h]
  _QWORD *v43; // [rsp+E8h] [rbp-48h]
  __int64 v44; // [rsp+F0h] [rbp-40h]
  _QWORD *v45; // [rsp+F8h] [rbp-38h]

  v28[0] = a2;
  if ( a2 )
  {
    v30.m128i_i64[0] = a2;
    v5 = (_QWORD *)a1[41];
    v30.m128i_i64[1] = a3;
    v7 = a1 + 40;
    if ( !v5 )
      goto LABEL_11;
    v8 = a1 + 40;
    do
    {
      while ( a2 <= v5[4] && (a2 != v5[4] || a3 <= v5[5]) )
      {
        v8 = v5;
        v5 = (_QWORD *)v5[2];
        if ( !v5 )
          goto LABEL_9;
      }
      v5 = (_QWORD *)v5[3];
    }
    while ( v5 );
LABEL_9:
    if ( v7 == v8 || a2 < v8[4] || a2 == v8[4] && a3 < v8[5] )
    {
LABEL_11:
      v31 = 0;
      v32 = 0;
      v33 = 0;
      v34 = 0;
      v37[0] = 0;
      v37[1] = 0;
      v38 = 0;
      v39 = 0;
      v40 = 0;
      v41 = 0;
      v42 = 0;
      v43 = 0;
      v44 = 0;
      v45 = 0;
      sub_1C55F90(v37, 0);
      v9 = v42;
      if ( v42 == (_QWORD *)(v44 - 8) )
      {
        sub_1C56080(v37, v28);
      }
      else
      {
        if ( v42 )
        {
          *v42 = v28[0];
          v9 = v42;
        }
        v42 = v9 + 1;
      }
      sub_1A64820((__int64)v35, (__int64)&v31, v28);
      while ( v42 != v38 )
      {
        if ( v42 == v43 )
        {
          v10 = *(_QWORD *)(*(v45 - 1) + 504LL);
          j_j___libc_free_0(v42, 512);
          v18 = *--v45 + 512LL;
          v43 = (_QWORD *)*v45;
          v44 = v18;
          v42 = v43 + 63;
        }
        else
        {
          v10 = *--v42;
        }
        v11 = sub_157EBA0(a3);
        v12 = sub_1C612E0(a1, a4, v10, v11);
        if ( !v12 )
        {
          v13 = *(_BYTE *)(v10 + 16);
          if ( (unsigned __int8)(v13 - 77) <= 1u || v13 == 54 )
          {
            v14 = (_QWORD *)a1[41];
            if ( v14 )
            {
              v15 = a1 + 40;
              do
              {
                while ( v14[4] >= v30.m128i_i64[0] && (v14[4] != v30.m128i_i64[0] || v14[5] >= v30.m128i_i64[1]) )
                {
                  v15 = v14;
                  v14 = (_QWORD *)v14[2];
                  if ( !v14 )
                    goto LABEL_29;
                }
                v14 = (_QWORD *)v14[3];
              }
              while ( v14 );
LABEL_29:
              if ( v7 == v15 || v15[4] > v30.m128i_i64[0] || v15[4] == v30.m128i_i64[0] && v15[5] > v30.m128i_i64[1] )
              {
LABEL_42:
                v35[0] = &v30;
                v19 = sub_1C56CE0(a1 + 39, v15, v35);
                v12 = 0;
                v15 = (_QWORD *)v19;
              }
              *((_BYTE *)v15 + 48) = 0;
              goto LABEL_33;
            }
            v15 = a1 + 40;
            goto LABEL_42;
          }
          if ( (*(_DWORD *)(v10 + 20) & 0xFFFFFFF) != 0 )
          {
            v25 = 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
            v20 = 0;
            do
            {
              if ( (*(_BYTE *)(v10 + 23) & 0x40) != 0 )
                v21 = *(_QWORD *)(v10 - 8);
              else
                v21 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
              v22 = *(_QWORD *)(v21 + v20);
              if ( *(_BYTE *)(v22 + 16) > 0x17u )
              {
                v29 = v22;
                v26 = v20;
                sub_1A64820((__int64)v35, (__int64)&v31, &v29);
                v20 = v26;
                if ( v36 )
                {
                  v23 = v42;
                  if ( v42 == (_QWORD *)(v44 - 8) )
                  {
                    sub_1C56080(v37, &v29);
                    v20 = v26;
                  }
                  else
                  {
                    if ( v42 )
                    {
                      *v42 = v29;
                      v23 = v42;
                    }
                    v42 = v23 + 1;
                  }
                }
              }
              v20 += 24;
            }
            while ( v25 != v20 );
          }
        }
      }
      v24 = (_BYTE *)sub_1C56DA0(a1 + 39, &v30);
      v12 = 1;
      *v24 = 1;
LABEL_33:
      v27 = v12;
      sub_1C55A80(v37);
      j___libc_free_0(v32);
      return v27;
    }
    else
    {
      return *((unsigned __int8 *)v8 + 48);
    }
  }
  else
  {
    return 1;
  }
}
