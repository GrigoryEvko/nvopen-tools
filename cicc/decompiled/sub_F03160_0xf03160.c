// Function: sub_F03160
// Address: 0xf03160
//
__int64 __fastcall sub_F03160(
        unsigned __int8 **a1,
        char *a2,
        unsigned __int64 *a3,
        unsigned __int64 a4,
        int a5,
        char a6)
{
  unsigned __int8 **v6; // r14
  unsigned __int64 *v7; // r13
  unsigned __int8 *v8; // rdi
  unsigned __int64 v9; // r12
  int v10; // r8d
  unsigned __int64 v11; // rbx
  unsigned __int16 v13; // r10
  __int64 v14; // r14
  unsigned int v15; // r8d
  unsigned int v17; // eax
  int v18; // r9d
  __int64 v19; // r10
  char v20; // r11
  int v21; // eax
  int v22; // eax
  int v23; // eax
  int v24; // eax
  unsigned __int8 *v25; // r11
  int v26; // eax
  int v27; // eax
  unsigned __int8 *v28; // rsi
  int v29; // eax
  int v30; // eax
  int v31; // eax
  int v32; // r9d
  unsigned int v33; // r9d
  int v34; // eax
  __int64 v35; // rcx
  char v36; // [rsp+Fh] [rbp-51h]
  unsigned __int8 **v38; // [rsp+18h] [rbp-48h]
  int v40; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = a3;
  v8 = *a1;
  v9 = *a3;
  if ( v8 < (unsigned __int8 *)a2 )
  {
    v10 = 0;
    v38 = v6;
    v11 = v9 + 4;
    v36 = a6 & 1;
    v40 = 0;
    while ( 1 )
    {
      v9 = v11 - 4;
      v13 = byte_3F88460[*v8];
      v14 = v13;
      if ( v13 >= a2 - (char *)v8 )
        break;
      if ( a4 <= v9 )
      {
        v6 = v38;
        v7 = a3;
        v15 = 2;
        goto LABEL_7;
      }
      if ( !(unsigned __int8)sub_F02F40(v8, (unsigned int)v13 + 1, (__int64)a3, a4, v10) )
      {
        if ( !a5 )
        {
          v6 = v38;
          v7 = a3;
          v15 = 3;
          goto LABEL_7;
        }
        goto LABEL_8;
      }
      switch ( v20 )
      {
        case 0:
          v31 = 0;
          goto LABEL_21;
        case 1:
          v28 = v8;
          v29 = 0;
          goto LABEL_20;
        case 2:
          v25 = v8;
          v26 = 0;
          goto LABEL_19;
        case 3:
          v23 = 0;
          goto LABEL_18;
        case 4:
          v21 = 0;
          goto LABEL_17;
        case 5:
          v34 = v18;
          v18 = *++v8;
          v21 = v34 << 6;
LABEL_17:
          v22 = v18 + v21;
          v18 = *++v8;
          v23 = v22 << 6;
LABEL_18:
          v24 = v18 + v23;
          v18 = v8[1];
          v25 = v8 + 1;
          v26 = v24 << 6;
LABEL_19:
          v27 = v18 + v26;
          v18 = v25[1];
          v28 = v25 + 1;
          v29 = v27 << 6;
LABEL_20:
          v8 = v28 + 1;
          v30 = v29 + v18;
          v18 = v28[1];
          v31 = v30 << 6;
LABEL_21:
          ++v8;
          v32 = v31 + v18;
          break;
        default:
          v32 = 0;
          break;
      }
      v33 = v32 - dword_3F88440[v19];
      if ( v33 > 0x10FFFF )
      {
        *(_DWORD *)(v11 - 4) = 65533;
        v9 = v11;
        v40 = 3;
      }
      else if ( v33 - 55296 > 0x7FF )
      {
        *(_DWORD *)(v11 - 4) = v33;
        v9 = v11;
      }
      else
      {
        if ( !a5 )
        {
          v35 = v14;
          v7 = a3;
          v6 = v38;
          v15 = 3;
          v8 += ~v35;
          goto LABEL_7;
        }
        *(_DWORD *)(v11 - 4) = 65533;
        v9 = v11;
      }
LABEL_9:
      v11 += 4LL;
      if ( v8 >= (unsigned __int8 *)a2 )
      {
        v15 = v40;
        v6 = v38;
        v7 = a3;
        goto LABEL_7;
      }
    }
    if ( !a5 || v36 )
    {
      v6 = v38;
      v7 = a3;
      v15 = 1;
      goto LABEL_7;
    }
LABEL_8:
    v9 = v11;
    v17 = sub_F03020((char *)v8, a2);
    *(_DWORD *)(v11 - 4) = 65533;
    v40 = 3;
    v8 += v17;
    goto LABEL_9;
  }
  v15 = 0;
LABEL_7:
  *v6 = v8;
  *v7 = v9;
  return v15;
}
