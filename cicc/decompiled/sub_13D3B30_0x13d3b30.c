// Function: sub_13D3B30
// Address: 0x13d3b30
//
__int64 __fastcall sub_13D3B30(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4, __int64 *a5)
{
  __int64 v8; // rax
  char v9; // dl
  unsigned __int64 v10; // rax
  unsigned int v11; // r9d
  char v12; // al
  __int64 v13; // r8
  __int64 *v15; // r9
  __int64 **v16; // r8
  __int64 v17; // rax
  __int64 v18; // r10
  __int64 v19; // rax
  _BYTE *v20; // rsi
  char v21; // al
  int v22; // r9d
  __int64 v23; // rsi
  int v24; // r9d
  int v25; // eax
  bool v26; // al
  __int64 v27; // [rsp+8h] [rbp-98h]
  int v28; // [rsp+8h] [rbp-98h]
  __int64 *v29; // [rsp+10h] [rbp-90h]
  int v30; // [rsp+10h] [rbp-90h]
  int v31; // [rsp+10h] [rbp-90h]
  bool v32; // [rsp+10h] [rbp-90h]
  __int64 v33; // [rsp+18h] [rbp-88h]
  __int64 v34; // [rsp+18h] [rbp-88h]
  __int64 **v35; // [rsp+18h] [rbp-88h]
  __int64 *v36; // [rsp+18h] [rbp-88h]
  __int64 v37; // [rsp+18h] [rbp-88h]
  __int64 v38; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v39; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v40; // [rsp+38h] [rbp-68h]
  __int64 *v41; // [rsp+40h] [rbp-60h] BYREF
  __int64 v42; // [rsp+48h] [rbp-58h]
  _BYTE v43[80]; // [rsp+50h] [rbp-50h] BYREF

  v8 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    v8 = *(_QWORD *)(v8 + 24);
  v9 = *(_BYTE *)(a2 + 16);
  if ( v9 != 9 && v9 != 15 )
  {
    if ( v9 )
      return 0;
    if ( (*(_BYTE *)(a2 + 33) & 0x20) != 0 )
    {
      v10 = 0xAAAAAAAAAAAAAAABLL * (a4 - a3);
      if ( (_DWORD)v10 )
      {
        v11 = *(_DWORD *)(a2 + 36);
        if ( (_DWORD)v10 == 1 )
        {
          v13 = sub_13CD740(v11, *a3, (__int64)a5);
          goto LABEL_13;
        }
        if ( (_DWORD)v10 == 2 )
        {
          v13 = sub_13CE480(a2, (_QWORD *)*a3, a3[3], a5);
          goto LABEL_13;
        }
        if ( v11 <= 0x67 )
        {
          if ( v11 > 0x65 )
          {
            v20 = (_BYTE *)a3[6];
            v30 = *(_DWORD *)(a2 + 36);
            v41 = &v38;
            v21 = sub_13D2630(&v41, v20);
            v22 = v30;
            if ( v21 )
            {
              v23 = v38;
              v40 = *(_DWORD *)(v38 + 8);
              if ( v40 > 0x40 )
              {
                sub_16A4EF0(&v39, v40, 0);
                v23 = v38;
                v22 = v30;
              }
              else
              {
                v39 = v40 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
              }
              v28 = v22;
              sub_16AB0A0(&v41, v23, &v39);
              v24 = v28;
              if ( (unsigned int)v42 <= 0x40 )
              {
                v26 = v41 == 0;
              }
              else
              {
                v31 = v42;
                v25 = sub_16A57B0(&v41);
                v24 = v28;
                v26 = v31 == v25;
                if ( v41 )
                {
                  v32 = v26;
                  j_j___libc_free_0_0(v41);
                  v26 = v32;
                  v24 = v28;
                }
              }
              if ( v26 )
              {
                v37 = a3[3 * (v24 != 102)];
                sub_135E100((__int64 *)&v39);
                v13 = v37;
                goto LABEL_13;
              }
              sub_135E100((__int64 *)&v39);
            }
          }
        }
        else if ( v11 == 129 )
        {
          v33 = a3[9];
          v12 = sub_13CB7C0((_BYTE *)a3[6]);
          v13 = v33;
          if ( v12 )
          {
LABEL_13:
            if ( v13 )
              return v13;
          }
        }
      }
    }
    if ( (unsigned __int8)sub_14D90D0(a1, a2) )
    {
      v15 = (__int64 *)v43;
      v16 = &v41;
      v41 = (__int64 *)v43;
      v42 = 0x400000000LL;
      if ( (unsigned __int64)((char *)a4 - (char *)a3) > 0x60 )
      {
        sub_16CD150(&v41, v43, 0xAAAAAAAAAAAAAAABLL * (a4 - a3), 8);
        v15 = (__int64 *)v43;
        v16 = &v41;
      }
      if ( a3 == a4 )
      {
LABEL_30:
        v36 = v15;
        v19 = sub_14DA350(a1);
        v15 = v36;
        v13 = v19;
      }
      else
      {
        while ( 1 )
        {
          v18 = *a3;
          if ( *(_BYTE *)(*a3 + 16) > 0x10u )
            break;
          v17 = (unsigned int)v42;
          if ( (unsigned int)v42 >= HIDWORD(v42) )
          {
            v27 = *a3;
            v29 = v15;
            v35 = v16;
            sub_16CD150(v16, v15, 0, 8);
            v17 = (unsigned int)v42;
            v18 = v27;
            v15 = v29;
            v16 = v35;
          }
          a3 += 3;
          v41[v17] = v18;
          LODWORD(v42) = v42 + 1;
          if ( a4 == a3 )
            goto LABEL_30;
        }
        v13 = 0;
      }
      if ( v41 != v15 )
      {
        v34 = v13;
        _libc_free((unsigned __int64)v41);
        return v34;
      }
      return v13;
    }
    return 0;
  }
  return sub_1599EF0(**(_QWORD **)(v8 + 16));
}
