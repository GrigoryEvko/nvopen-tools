// Function: sub_16F08E0
// Address: 0x16f08e0
//
__int64 __fastcall sub_16F08E0(
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
  unsigned int v32; // r9d
  int v33; // eax
  __int64 v34; // rcx
  char v35; // [rsp+Fh] [rbp-51h]
  unsigned __int8 **v37; // [rsp+18h] [rbp-48h]
  int v39; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = a3;
  v8 = *a1;
  v9 = *a3;
  if ( v8 >= (unsigned __int8 *)a2 )
  {
    v15 = 0;
    goto LABEL_7;
  }
  v10 = 0;
  v37 = v6;
  v11 = v9 + 4;
  v35 = a6 & 1;
  v39 = 0;
  while ( 1 )
  {
    v9 = v11 - 4;
    v13 = byte_42AFA40[*v8];
    v14 = v13;
    if ( v13 >= a2 - (char *)v8 )
    {
      if ( !a5 || v35 )
      {
        v6 = v37;
        v7 = a3;
        v15 = 1;
        goto LABEL_7;
      }
      goto LABEL_8;
    }
    if ( a4 <= v9 )
      break;
    if ( (unsigned __int8)sub_16F06C0(v8, (unsigned int)v13 + 1, (__int64)a3, a4, v10) )
    {
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
          v33 = v18;
          v18 = *++v8;
          v21 = v33 << 6;
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
          v32 = v31 + v18 - dword_42AFA10[v19];
          if ( v32 > 0x10FFFF )
          {
            *(_DWORD *)(v11 - 4) = 65533;
            v9 = v11;
            v39 = 3;
          }
          else if ( v32 - 55296 > 0x7FF )
          {
            *(_DWORD *)(v11 - 4) = v32;
            v9 = v11;
          }
          else
          {
            if ( !a5 )
            {
              v34 = v14;
              v7 = a3;
              v6 = v37;
              v15 = 3;
              v8 += ~v34;
              goto LABEL_7;
            }
            *(_DWORD *)(v11 - 4) = 65533;
            v9 = v11;
          }
          break;
        default:
          JUMPOUT(0x41A81C);
      }
      goto LABEL_9;
    }
    if ( !a5 )
    {
      v6 = v37;
      v7 = a3;
      v15 = 3;
      goto LABEL_7;
    }
LABEL_8:
    v9 = v11;
    v17 = sub_16F07A0((char *)v8, a2);
    *(_DWORD *)(v11 - 4) = 65533;
    v39 = 3;
    v8 += v17;
LABEL_9:
    v11 += 4LL;
    if ( v8 >= (unsigned __int8 *)a2 )
    {
      v15 = v39;
      v6 = v37;
      v7 = a3;
      goto LABEL_7;
    }
  }
  v6 = v37;
  v7 = a3;
  v15 = 2;
LABEL_7:
  *v6 = v8;
  *v7 = v9;
  return v15;
}
