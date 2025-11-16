// Function: sub_1515D60
// Address: 0x1515d60
//
__int64 *__fastcall sub_1515D60(
        __int64 *a1,
        __int64 a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 a6,
        void (__fastcall *a7)(__int64, __int64, _QWORD),
        __int64 a8)
{
  unsigned __int64 v8; // r14
  unsigned int v9; // ebx
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // r12
  const char *v12; // rax
  unsigned int v14; // r10d
  unsigned __int64 *v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned int v17; // r9d
  unsigned __int64 v18; // rax
  char v19; // cl
  unsigned __int64 v20; // rax
  unsigned int v21; // edi
  char v22; // r10
  unsigned int v23; // r15d
  unsigned __int64 *v24; // r8
  unsigned __int64 v25; // rsi
  unsigned int v26; // r11d
  unsigned __int64 v27; // rax
  char v28; // cl
  unsigned __int64 v29; // rax
  char v30; // al
  unsigned int v31; // r11d
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rdx
  char v35; // cl
  char v36; // al
  unsigned int v37; // r9d
  __int64 v38; // r8
  __int64 v39; // rax
  __int64 v40; // rdx
  char v41; // cl
  int v42; // [rsp+4h] [rbp-8Ch]
  __int64 v44; // [rsp+20h] [rbp-70h]
  __int64 v45; // [rsp+28h] [rbp-68h]
  unsigned __int64 v46; // [rsp+30h] [rbp-60h]
  _QWORD v48[2]; // [rsp+40h] [rbp-50h] BYREF
  char v49; // [rsp+50h] [rbp-40h]
  char v50; // [rsp+51h] [rbp-3Fh]

  if ( a4 != 2 )
  {
    v50 = 1;
    v48[0] = "Invalid record: metadata strings layout";
    v49 = 3;
    sub_1514BE0(a1, (__int64)v48);
    return a1;
  }
  v42 = *a3;
  if ( !v42 )
  {
    v50 = 1;
    v12 = "Invalid record: metadata strings with no strings";
    goto LABEL_8;
  }
  v8 = (unsigned int)a3[1];
  v44 = a3[1];
  if ( a6 < v8 )
  {
    v50 = 1;
    v12 = "Invalid record: metadata strings corrupt offset";
    goto LABEL_8;
  }
  v9 = 0;
  v10 = 0;
  v46 = a6 - v8;
  v11 = 0;
  v45 = v8 + a5;
  while ( 1 )
  {
    if ( !v9 )
    {
      if ( v8 <= v11 )
      {
        v50 = 1;
        v12 = "Invalid record: metadata strings bad length";
        goto LABEL_8;
      }
      v10 = 0;
      v14 = 6;
      goto LABEL_12;
    }
    if ( v9 <= 5 )
    {
      v14 = 6 - v9;
      if ( v8 <= v11 )
        goto LABEL_32;
LABEL_12:
      v15 = (unsigned __int64 *)(a5 + v11);
      if ( v8 < v11 + 8 )
      {
        v37 = v44 - v11;
        if ( (_DWORD)v44 == (_DWORD)v11 )
          goto LABEL_32;
        v38 = v37;
        v39 = 0;
        v16 = 0;
        do
        {
          v40 = *((unsigned __int8 *)v15 + v39);
          v41 = 8 * v39++;
          v16 |= v40 << v41;
        }
        while ( v37 != v39 );
        v17 = 8 * v37;
        v11 += v38;
        if ( v17 < v14 )
          goto LABEL_32;
      }
      else
      {
        v16 = *v15;
        v11 += 8LL;
        v17 = 64;
      }
      v18 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 + 58);
      v19 = v9;
      v9 = v9 + v17 - 6;
      v20 = v10 | ((v16 & v18) << v19);
      v10 = v16 >> v14;
      goto LABEL_15;
    }
    v36 = v10;
    v9 -= 6;
    v10 >>= 6;
    LODWORD(v20) = v36 & 0x3F;
LABEL_15:
    v21 = v20;
    if ( (v20 & 0x20) != 0 )
    {
      v21 = v20 & 0x1F;
      v22 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v22 += 5;
          if ( v9 <= 5 )
            break;
          v30 = v10;
          v10 >>= 6;
          v9 -= 6;
          v30 &= 0x3Fu;
          v21 |= (v30 & 0x1F) << v22;
          if ( (v30 & 0x20) == 0 )
            goto LABEL_25;
        }
        if ( !v9 )
          v10 = 0;
        v23 = 6 - v9;
        if ( v8 <= v11 )
          break;
        v24 = (unsigned __int64 *)(a5 + v11);
        if ( v8 < v11 + 8 )
        {
          v31 = v44 - v11;
          if ( (_DWORD)v44 == (_DWORD)v11 )
            break;
          v32 = v31;
          v33 = 0;
          v25 = 0;
          do
          {
            v34 = *((unsigned __int8 *)v24 + v33);
            v35 = 8 * v33++;
            v25 |= v34 << v35;
          }
          while ( v31 != v33 );
          v26 = 8 * v31;
          v11 += v32;
          if ( v23 > v26 )
            break;
        }
        else
        {
          v25 = *v24;
          v11 += 8LL;
          v26 = 64;
        }
        v27 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v9 + 58);
        v28 = v9;
        v9 = v9 + v26 - 6;
        v29 = v10 | ((v25 & v27) << v28);
        v10 = v25 >> v23;
        v21 |= (v29 & 0x1F) << v22;
        if ( (v29 & 0x20) == 0 )
          goto LABEL_25;
      }
LABEL_32:
      sub_16BD130("Unexpected end of file", 1);
    }
LABEL_25:
    if ( v21 > v46 )
      break;
    a7(a8, v45, v21);
    v46 -= v21;
    v45 += v21;
    if ( !--v42 )
    {
      *a1 = 1;
      return a1;
    }
  }
  v50 = 1;
  v12 = "Invalid record: metadata strings truncated chars";
LABEL_8:
  v48[0] = v12;
  v49 = 3;
  sub_1514BE0(a1, (__int64)v48);
  return a1;
}
