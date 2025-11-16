// Function: sub_1C2FC50
// Address: 0x1c2fc50
//
__int64 __fastcall sub_1C2FC50(__int64 a1)
{
  __int64 v1; // rdi
  __int64 v2; // rbx
  unsigned int v3; // r12d
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r15
  _BYTE *v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 result; // rax
  __int64 v14; // rbx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rax
  int v19; // [rsp+10h] [rbp-F0h]
  int v20; // [rsp+14h] [rbp-ECh]
  __int64 v21; // [rsp+18h] [rbp-E8h]
  char *v22; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v23; // [rsp+30h] [rbp-D0h]
  _BYTE *v24; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v25; // [rsp+48h] [rbp-B8h]
  _BYTE v26[176]; // [rsp+50h] [rbp-B0h] BYREF

  v24 = v26;
  v25 = 0x1000000000LL;
  v23 = 257;
  v1 = *(_QWORD *)(a1 + 40);
  if ( *off_4CD4988 )
  {
    v22 = off_4CD4988;
    LOBYTE(v23) = 3;
  }
  v2 = sub_1632310(v1, (__int64)&v22);
  if ( !v2 )
    goto LABEL_18;
  v3 = 0;
  v19 = v25;
  v20 = sub_161F520(v2);
  if ( v20 )
  {
    while ( 1 )
    {
      v4 = sub_161F530(v2, v3);
      v5 = *(unsigned int *)(v4 + 8);
      v6 = *(_QWORD *)(v4 - 8 * v5);
      if ( v6 )
      {
        if ( *(_BYTE *)v6 == 1 )
        {
          v7 = *(_QWORD *)(v6 + 136);
          if ( *(_BYTE *)(v7 + 16) <= 3u && a1 == v7 && (unsigned int)v5 > 1 )
            break;
        }
      }
LABEL_16:
      if ( v20 == ++v3 )
        goto LABEL_17;
    }
    v8 = (unsigned int)v5;
    v9 = 1;
    while ( 1 )
    {
      v10 = *(_BYTE **)(v4 + 8 * (v9 - v8));
      if ( *v10 )
        v10 = 0;
      v11 = sub_161E970((__int64)v10);
      if ( v12 == 14
        && *(_QWORD *)v11 == 0x685F746978657461LL
        && *(_DWORD *)(v11 + 8) == 1818521185
        && *(_WORD *)(v11 + 12) == 29285 )
      {
        break;
      }
      v9 += 2;
      if ( (unsigned int)v5 <= (unsigned int)v9 )
        goto LABEL_16;
      v8 = *(unsigned int *)(v4 + 8);
    }
    v14 = sub_1C2E3F0(v4 + 8 * ((unsigned int)(v9 + 1) - (unsigned __int64)*(unsigned int *)(v4 + 8)));
    v17 = (unsigned int)v25;
    if ( (unsigned int)v25 >= HIDWORD(v25) )
    {
      sub_16CD150((__int64)&v24, v26, 0, 8, v15, v16);
      v17 = (unsigned int)v25;
    }
    *(_QWORD *)&v24[8 * v17] = v14;
    LODWORD(v25) = v25 + 1;
    goto LABEL_27;
  }
LABEL_17:
  if ( v19 == (_DWORD)v25 )
  {
LABEL_18:
    if ( v24 == v26 )
      return 0;
    _libc_free((unsigned __int64)v24);
    return 0;
  }
LABEL_27:
  result = *(_QWORD *)v24;
  if ( v24 != v26 )
  {
    v21 = *(_QWORD *)v24;
    _libc_free((unsigned __int64)v24);
    return v21;
  }
  return result;
}
