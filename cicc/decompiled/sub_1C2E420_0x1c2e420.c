// Function: sub_1C2E420
// Address: 0x1c2e420
//
__int64 __fastcall sub_1C2E420(__int64 a1, const void *a2, size_t a3, _QWORD *a4)
{
  __int64 v4; // rdi
  __int64 v5; // r13
  unsigned int v6; // r14d
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  _BYTE *v13; // rdi
  const void *v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rdi
  unsigned int v17; // r12d
  _BYTE *v19; // rbx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  int v23; // [rsp+4h] [rbp-11Ch]
  __int64 v24; // [rsp+8h] [rbp-118h]
  int v28; // [rsp+34h] [rbp-ECh]
  char *v29; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v30; // [rsp+50h] [rbp-D0h]
  _QWORD *v31; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+68h] [rbp-B8h]
  _BYTE v33[176]; // [rsp+70h] [rbp-B0h] BYREF

  v31 = v33;
  v32 = 0x1000000000LL;
  v30 = 257;
  v4 = *(_QWORD *)(a1 + 40);
  if ( *off_4CD4988 )
  {
    v29 = off_4CD4988;
    LOBYTE(v30) = 3;
  }
  v5 = sub_1632310(v4, (__int64)&v29);
  if ( !v5 )
    goto LABEL_19;
  v6 = 0;
  v23 = v32;
  v28 = sub_161F520(v5);
  if ( v28 )
  {
    while ( 1 )
    {
      v7 = sub_161F530(v5, v6);
      v8 = *(unsigned int *)(v7 + 8);
      v9 = *(_QWORD *)(v7 - 8 * v8);
      if ( v9 )
      {
        if ( *(_BYTE *)v9 == 1 )
        {
          v10 = *(_QWORD *)(v9 + 136);
          if ( *(_BYTE *)(v10 + 16) <= 3u && a1 == v10 && (unsigned int)v8 > 1 )
            break;
        }
      }
LABEL_17:
      if ( v28 == ++v6 )
        goto LABEL_18;
    }
    v11 = (unsigned int)v8;
    v24 = v5;
    v12 = 1;
    while ( 1 )
    {
      v13 = *(_BYTE **)(v7 + 8 * (v12 - v11));
      if ( *v13 )
        v13 = 0;
      v14 = (const void *)sub_161E970((__int64)v13);
      if ( a3 == v15 && (!a3 || !memcmp(v14, a2, a3)) )
        break;
      v12 += 2;
      if ( (unsigned int)v8 <= (unsigned int)v12 )
      {
        v5 = v24;
        goto LABEL_17;
      }
      v11 = *(unsigned int *)(v7 + 8);
    }
    v19 = sub_1C2E400((_QWORD *)(v7 + 8 * ((unsigned int)(v12 + 1) - (unsigned __int64)*(unsigned int *)(v7 + 8))));
    v22 = (unsigned int)v32;
    if ( (unsigned int)v32 >= HIDWORD(v32) )
    {
      sub_16CD150((__int64)&v31, v33, 0, 8, v20, v21);
      v22 = (unsigned int)v32;
    }
    v31[v22] = v19;
    LODWORD(v32) = v32 + 1;
    goto LABEL_28;
  }
LABEL_18:
  if ( v23 == (_DWORD)v32 )
  {
LABEL_19:
    v16 = v31;
    v17 = 0;
    goto LABEL_20;
  }
LABEL_28:
  v16 = v31;
  v17 = 1;
  *a4 = *v31;
LABEL_20:
  if ( v16 != v33 )
    _libc_free((unsigned __int64)v16);
  return v17;
}
