// Function: sub_2C9EA10
// Address: 0x2c9ea10
//
__int64 __fastcall sub_2C9EA10(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // r13
  _QWORD *v5; // rax
  __int64 v7; // r15
  _QWORD *v8; // rdx
  _QWORD *v9; // rax
  unsigned __int8 *v10; // r12
  unsigned __int64 v11; // rax
  int v12; // edx
  __int64 v13; // rcx
  __int64 v14; // rax
  int v15; // edx
  unsigned int v16; // eax
  unsigned int v17; // r12d
  _QWORD *v18; // rax
  __int64 v19; // rsi
  __int64 v21; // rdx
  __int64 v22; // r13
  unsigned __int8 *v23; // rsi
  _QWORD *v24; // rdx
  unsigned __int64 v25; // [rsp+10h] [rbp-140h]
  __int64 v26; // [rsp+28h] [rbp-128h]
  unsigned __int64 v27; // [rsp+50h] [rbp-100h]
  __int64 v28[2]; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v29; // [rsp+68h] [rbp-E8h] BYREF
  __m128i v30; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+88h] [rbp-C8h]
  __int64 v33; // [rsp+90h] [rbp-C0h]
  __int64 v34; // [rsp+98h] [rbp-B8h]
  const __m128i *v35[4]; // [rsp+A0h] [rbp-B0h] BYREF
  char v36; // [rsp+C0h] [rbp-90h]
  __int64 v37[2]; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD *v38; // [rsp+E0h] [rbp-70h]
  __int64 v39; // [rsp+E8h] [rbp-68h]
  __int64 v40; // [rsp+F0h] [rbp-60h]
  __int64 v41; // [rsp+F8h] [rbp-58h]
  _QWORD *v42; // [rsp+100h] [rbp-50h]
  _QWORD *v43; // [rsp+108h] [rbp-48h]
  __int64 v44; // [rsp+110h] [rbp-40h]
  _QWORD *v45; // [rsp+118h] [rbp-38h]

  v28[0] = a2;
  if ( a2 )
  {
    v30.m128i_i64[0] = a2;
    v4 = a3;
    v5 = *(_QWORD **)(a1 + 328);
    v30.m128i_i64[1] = a3;
    v7 = a1 + 320;
    if ( !v5 )
      goto LABEL_11;
    v8 = (_QWORD *)(a1 + 320);
    do
    {
      while ( a2 <= v5[4] && (a2 != v5[4] || v4 <= v5[5]) )
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
    if ( (_QWORD *)v7 == v8 || a2 < v8[4] || a2 == v8[4] && v4 < v8[5] )
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
      sub_2785050(v37, 0);
      v9 = v42;
      if ( v42 == (_QWORD *)(v44 - 8) )
      {
        sub_2785520((unsigned __int64 *)v37, v28);
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
      sub_2400480((__int64)v35, (__int64)&v31, v28);
      v27 = v4 + 48;
      while ( v42 != v38 )
      {
        if ( v42 == v43 )
        {
          v10 = *(unsigned __int8 **)(*(v45 - 1) + 504LL);
          j_j___libc_free_0((unsigned __int64)v42);
          v21 = *--v45 + 512LL;
          v43 = (_QWORD *)*v45;
          v44 = v21;
          v42 = v43 + 63;
        }
        else
        {
          v10 = (unsigned __int8 *)*--v42;
        }
        v11 = *(_QWORD *)(v4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v11 == v27 )
        {
          v13 = 0;
        }
        else
        {
          if ( !v11 )
            BUG();
          v12 = *(unsigned __int8 *)(v11 - 24);
          v13 = 0;
          v14 = v11 - 24;
          if ( (unsigned int)(v12 - 30) < 0xB )
            v13 = v14;
        }
        if ( !(unsigned __int8)sub_2C9D230(a1, a4, (__int64)v10, v13) )
        {
          v15 = *v10;
          v16 = v15 - 84;
          LOBYTE(v16) = (_BYTE)v15 != 61 && (unsigned __int8)(v15 - 84) > 1u;
          if ( !(_BYTE)v16 )
          {
            v17 = v16;
            v18 = *(_QWORD **)(a1 + 328);
            if ( v18 )
            {
              v19 = a1 + 320;
              do
              {
                while ( v18[4] >= v30.m128i_i64[0] && (v18[4] != v30.m128i_i64[0] || v18[5] >= v30.m128i_i64[1]) )
                {
                  v19 = (__int64)v18;
                  v18 = (_QWORD *)v18[2];
                  if ( !v18 )
                    goto LABEL_32;
                }
                v18 = (_QWORD *)v18[3];
              }
              while ( v18 );
LABEL_32:
              if ( v7 == v19
                || *(_QWORD *)(v19 + 32) > v30.m128i_i64[0]
                || *(_QWORD *)(v19 + 32) == v30.m128i_i64[0] && *(_QWORD *)(v19 + 40) > v30.m128i_i64[1] )
              {
LABEL_45:
                v35[0] = &v30;
                v19 = sub_2C96A50((_QWORD *)(a1 + 312), (_QWORD *)v19, v35);
              }
              *(_BYTE *)(v19 + 48) = 0;
              goto LABEL_36;
            }
            v19 = a1 + 320;
            goto LABEL_45;
          }
          if ( (*((_DWORD *)v10 + 1) & 0x7FFFFFF) != 0 )
          {
            v26 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
            v25 = v4;
            v22 = 0;
            do
            {
              if ( (v10[7] & 0x40) != 0 )
                v23 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
              else
                v23 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
              if ( **(_BYTE **)&v23[v22] > 0x1Cu )
              {
                v29 = *(_QWORD *)&v23[v22];
                sub_2400480((__int64)v35, (__int64)&v31, &v29);
                if ( v36 )
                {
                  v24 = v42;
                  if ( v42 == (_QWORD *)(v44 - 8) )
                  {
                    sub_2785520((unsigned __int64 *)v37, &v29);
                  }
                  else
                  {
                    if ( v42 )
                    {
                      *v42 = v29;
                      v24 = v42;
                    }
                    v42 = v24 + 1;
                  }
                }
              }
              v22 += 32;
            }
            while ( v26 != v22 );
            v4 = v25;
          }
        }
      }
      v17 = 1;
      *(_BYTE *)sub_2C96B10((_QWORD *)(a1 + 312), &v30) = 1;
LABEL_36:
      sub_2784FD0((unsigned __int64 *)v37);
      sub_C7D6A0(v32, 8LL * (unsigned int)v34, 8);
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
  return v17;
}
