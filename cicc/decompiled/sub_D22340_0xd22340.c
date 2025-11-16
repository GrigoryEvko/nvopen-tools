// Function: sub_D22340
// Address: 0xd22340
//
__int64 __fastcall sub_D22340(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _BYTE *v7; // r13
  __int64 result; // rax
  __int64 v9; // r8
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 *v13; // rdx
  char v14; // dl
  unsigned __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // [rsp+8h] [rbp-A8h]
  __int64 v20; // [rsp+8h] [rbp-A8h]
  __int64 *v21; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v22; // [rsp+18h] [rbp-98h]
  unsigned int v23; // [rsp+1Ch] [rbp-94h]
  _QWORD v24[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v25; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v26; // [rsp+48h] [rbp-68h]
  __int64 v27; // [rsp+50h] [rbp-60h]
  int v28; // [rsp+58h] [rbp-58h]
  char v29; // [rsp+5Ch] [rbp-54h]
  _BYTE *v30; // [rsp+60h] [rbp-50h] BYREF

  v6 = a2;
  v7 = a1;
  v26 = (__int64 *)&v30;
  v27 = 0x100000004LL;
  LODWORD(result) = 1;
  v21 = v24;
  v23 = 4;
  v24[0] = a1;
  v28 = 0;
  v29 = 1;
  v30 = a1;
  v25 = 1;
  while ( 1 )
  {
    v22 = result - 1;
    if ( *v7 != 57 || (v10 = *((_QWORD *)v7 - 8)) == 0 || (v11 = *((_QWORD *)v7 - 4)) == 0 )
    {
      if ( !sub_D22300((__int64)v7) )
      {
        v16 = *(unsigned int *)(v6 + 8);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 12) )
        {
          a2 = v6 + 16;
          sub_C8D5F0(v6, (const void *)(v6 + 16), v16 + 1, 8u, v9, a6);
          v16 = *(unsigned int *)(v6 + 8);
        }
        *(_QWORD *)(*(_QWORD *)v6 + 8 * v16) = v7;
        ++*(_DWORD *)(v6 + 8);
      }
LABEL_3:
      result = v22;
      goto LABEL_4;
    }
    if ( !v29 )
      goto LABEL_12;
    v12 = v26;
    a2 = HIDWORD(v27);
    a4 = (__int64)&v26[HIDWORD(v27)];
    a3 = (__int64)v26;
    if ( v26 == (__int64 *)a4 )
    {
LABEL_11:
      if ( HIDWORD(v27) < (unsigned int)v27 )
      {
        ++HIDWORD(v27);
        *(_QWORD *)a4 = v10;
        ++v25;
      }
      else
      {
LABEL_12:
        v19 = *((_QWORD *)v7 - 8);
        sub_C8CC70((__int64)&v25, v19, a3, a4, v10, a6);
        v10 = v19;
        if ( !(_BYTE)v13 )
        {
          if ( !v29 )
            goto LABEL_14;
LABEL_37:
          v12 = v26;
          a2 = HIDWORD(v27);
          goto LABEL_24;
        }
      }
      v17 = v22;
      a4 = v23;
      v18 = v22 + 1LL;
      if ( v18 > v23 )
      {
        v20 = v10;
        sub_C8D5F0((__int64)&v21, v24, v18, 8u, v10, a6);
        v17 = v22;
        v10 = v20;
      }
      v13 = v21;
      v21[v17] = v10;
      ++v22;
      if ( !v29 )
        goto LABEL_14;
      goto LABEL_37;
    }
    while ( v10 != *(_QWORD *)a3 )
    {
      a3 += 8;
      if ( a4 == a3 )
        goto LABEL_11;
    }
LABEL_24:
    v13 = &v12[(unsigned int)a2];
    if ( v12 != v13 )
    {
      while ( v11 != *v12 )
      {
        if ( v13 == ++v12 )
          goto LABEL_27;
      }
      goto LABEL_3;
    }
LABEL_27:
    if ( (unsigned int)a2 < (unsigned int)v27 )
      break;
LABEL_14:
    a2 = v11;
    sub_C8CC70((__int64)&v25, v11, (__int64)v13, a4, v10, a6);
    result = v22;
    if ( v14 )
    {
      v15 = v22 + 1LL;
      if ( v15 <= v23 )
        goto LABEL_16;
      goto LABEL_38;
    }
LABEL_4:
    if ( !(_DWORD)result )
      goto LABEL_17;
LABEL_5:
    a4 = (__int64)v21;
    a3 = (unsigned int)result;
    v7 = (_BYTE *)v21[(unsigned int)result - 1];
  }
  a2 = (unsigned int)(a2 + 1);
  HIDWORD(v27) = a2;
  *v13 = v11;
  result = v22;
  ++v25;
  v15 = v22 + 1LL;
  if ( v15 <= v23 )
    goto LABEL_16;
LABEL_38:
  a2 = (__int64)v24;
  sub_C8D5F0((__int64)&v21, v24, v15, 8u, v10, a6);
  result = v22;
LABEL_16:
  v21[result] = v11;
  result = v22 + 1;
  v22 = result;
  if ( (_DWORD)result )
    goto LABEL_5;
LABEL_17:
  if ( !v29 )
    result = _libc_free(v26, a2);
  if ( v21 != v24 )
    return _libc_free(v21, a2);
  return result;
}
