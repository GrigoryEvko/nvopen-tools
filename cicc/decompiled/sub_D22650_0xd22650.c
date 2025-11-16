// Function: sub_D22650
// Address: 0xd22650
//
__int64 __fastcall sub_D22650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 *v7; // rbx
  __int64 result; // rax
  __int64 v9; // r8
  __int64 v10; // r15
  __int64 *v11; // rax
  __int64 *v12; // rdx
  char v13; // dl
  unsigned __int64 v14; // rdx
  bool v15; // zf
  __int64 v16; // rax
  unsigned __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 v19; // [rsp+8h] [rbp-A8h]
  __int64 *v20; // [rsp+10h] [rbp-A0h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-98h]
  unsigned int v22; // [rsp+1Ch] [rbp-94h]
  _QWORD v23[4]; // [rsp+20h] [rbp-90h] BYREF
  __int64 v24; // [rsp+40h] [rbp-70h] BYREF
  __int64 *v25; // [rsp+48h] [rbp-68h]
  __int64 v26; // [rsp+50h] [rbp-60h]
  int v27; // [rsp+58h] [rbp-58h]
  char v28; // [rsp+5Ch] [rbp-54h]
  __int64 v29; // [rsp+60h] [rbp-50h] BYREF

  v6 = a1;
  v7 = (__int64 *)a2;
  v25 = &v29;
  v26 = 0x100000004LL;
  LODWORD(result) = 1;
  v20 = v23;
  v22 = 4;
  v23[0] = a1;
  v27 = 0;
  v28 = 1;
  v29 = a1;
  v24 = 1;
  while ( 1 )
  {
    v21 = result - 1;
    if ( *(_BYTE *)v6 != 57 )
      break;
    v9 = *(_QWORD *)(v6 - 64);
    if ( !v9 )
      break;
    v10 = *(_QWORD *)(v6 - 32);
    if ( !v10 )
      break;
    if ( !v28 )
      goto LABEL_13;
    v11 = v25;
    a2 = HIDWORD(v26);
    a4 = (__int64)&v25[HIDWORD(v26)];
    a3 = (__int64)v25;
    if ( v25 == (__int64 *)a4 )
    {
LABEL_12:
      if ( HIDWORD(v26) < (unsigned int)v26 )
      {
        ++HIDWORD(v26);
        *(_QWORD *)a4 = v9;
        ++v24;
      }
      else
      {
LABEL_13:
        v18 = *(_QWORD *)(v6 - 64);
        sub_C8CC70((__int64)&v24, v18, a3, a4, v9, a6);
        v9 = v18;
        if ( !(_BYTE)v12 )
        {
          if ( !v28 )
            goto LABEL_15;
LABEL_37:
          v11 = v25;
          a2 = HIDWORD(v26);
          goto LABEL_24;
        }
      }
      v16 = v21;
      a4 = v22;
      v17 = v21 + 1LL;
      if ( v17 > v22 )
      {
        v19 = v9;
        sub_C8D5F0((__int64)&v20, v23, v17, 8u, v9, a6);
        v16 = v21;
        v9 = v19;
      }
      v12 = v20;
      v20[v16] = v9;
      ++v21;
      if ( !v28 )
        goto LABEL_15;
      goto LABEL_37;
    }
    while ( v9 != *(_QWORD *)a3 )
    {
      a3 += 8;
      if ( a4 == a3 )
        goto LABEL_12;
    }
LABEL_24:
    v12 = &v11[(unsigned int)a2];
    if ( v11 != v12 )
    {
      while ( v10 != *v11 )
      {
        if ( v12 == ++v11 )
          goto LABEL_27;
      }
      goto LABEL_4;
    }
LABEL_27:
    if ( (unsigned int)a2 < (unsigned int)v26 )
    {
      a2 = (unsigned int)(a2 + 1);
      HIDWORD(v26) = a2;
      *v12 = v10;
      result = v21;
      ++v24;
      v14 = v21 + 1LL;
      if ( v14 <= v22 )
        goto LABEL_17;
LABEL_38:
      a2 = (__int64)v23;
      sub_C8D5F0((__int64)&v20, v23, v14, 8u, v9, a6);
      result = v21;
      goto LABEL_17;
    }
LABEL_15:
    a2 = v10;
    sub_C8CC70((__int64)&v24, v10, (__int64)v12, a4, v9, a6);
    result = v21;
    if ( !v13 )
      goto LABEL_5;
    v14 = v21 + 1LL;
    if ( v14 > v22 )
      goto LABEL_38;
LABEL_17:
    v20[result] = v10;
    result = v21 + 1;
    v21 = result;
    if ( !(_DWORD)result )
    {
LABEL_18:
      if ( v28 )
        goto LABEL_19;
      goto LABEL_32;
    }
LABEL_6:
    a4 = (__int64)v20;
    a3 = (unsigned int)result;
    v6 = v20[(unsigned int)result - 1];
  }
  if ( !sub_D22300(v6) || (result = *(_QWORD *)(v6 + 16)) == 0 || *(_QWORD *)(result + 8) )
  {
LABEL_4:
    result = v21;
LABEL_5:
    if ( !(_DWORD)result )
      goto LABEL_18;
    goto LABEL_6;
  }
  v15 = v28 == 0;
  *v7 = v6;
  if ( !v15 )
    goto LABEL_19;
LABEL_32:
  result = _libc_free(v25, a2);
LABEL_19:
  if ( v20 != v23 )
    return _libc_free(v20, a2);
  return result;
}
