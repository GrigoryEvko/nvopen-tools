// Function: sub_1B7E640
// Address: 0x1b7e640
//
void __fastcall sub_1B7E640(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v6; // r8
  __int64 *v8; // r13
  unsigned int v9; // ecx
  __int64 v10; // rax
  bool v11; // zf
  __int64 v12; // r14
  __int64 v13; // r15
  unsigned __int8 v14; // dl
  _QWORD **v15; // r13
  _BYTE *v16; // r14
  int v17; // edx
  __int64 v18; // rsi
  unsigned int v19; // eax
  _QWORD *v20; // rcx
  int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rdx
  int v24; // edx
  int v25; // r8d
  __int64 *v26; // [rsp+8h] [rbp-C8h]
  __int64 *v27; // [rsp+8h] [rbp-C8h]
  _BYTE *v28; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v29; // [rsp+18h] [rbp-B8h]
  _BYTE v30[176]; // [rsp+20h] [rbp-B0h] BYREF

  v6 = &a2[a3];
  v28 = v30;
  v29 = 0x1000000000LL;
  if ( a2 == v6 )
    return;
  v8 = a2;
  v9 = 16;
  v10 = 0;
  while ( 1 )
  {
    v12 = *v8;
    v13 = 0;
    v14 = *(_BYTE *)(*v8 + 16);
    if ( v14 > 0x17u )
    {
      if ( v14 == 54 || v14 == 55 )
      {
        v13 = *(_QWORD *)(v12 - 24);
        if ( (unsigned int)v10 < v9 )
          goto LABEL_6;
        goto LABEL_12;
      }
      if ( v14 == 78 )
      {
        v23 = *(_QWORD *)(v12 - 24);
        if ( !*(_BYTE *)(v23 + 16) )
        {
          v24 = *(_DWORD *)(v23 + 36);
          if ( v24 == 4085 || v24 == 4057 )
          {
            v13 = *(_QWORD *)(v12 + 24 * (1LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
          }
          else if ( v24 == 4503 || v24 == 4492 )
          {
            v13 = *(_QWORD *)(v12 + 24 * (2LL - (*(_DWORD *)(v12 + 20) & 0xFFFFFFF)));
          }
        }
      }
    }
    if ( (unsigned int)v10 < v9 )
      goto LABEL_6;
LABEL_12:
    v26 = v6;
    sub_16CD150((__int64)&v28, v30, 0, 8, (int)v6, a6);
    v10 = (unsigned int)v29;
    v6 = v26;
LABEL_6:
    *(_QWORD *)&v28[8 * v10] = v12;
    v10 = (unsigned int)(v29 + 1);
    v11 = *(_BYTE *)(v13 + 16) == 56;
    LODWORD(v29) = v29 + 1;
    if ( v11 )
      break;
    if ( v6 == ++v8 )
      goto LABEL_16;
LABEL_8:
    v9 = HIDWORD(v29);
  }
  if ( (unsigned int)v10 >= HIDWORD(v29) )
  {
    v27 = v6;
    sub_16CD150((__int64)&v28, v30, 0, 8, (int)v6, a6);
    v10 = (unsigned int)v29;
    v6 = v27;
  }
  ++v8;
  *(_QWORD *)&v28[8 * v10] = v13;
  v10 = (unsigned int)(v29 + 1);
  LODWORD(v29) = v29 + 1;
  if ( v6 != v8 )
    goto LABEL_8;
LABEL_16:
  v15 = (_QWORD **)v28;
  v16 = &v28[8 * v10];
  if ( v28 == v16 )
    goto LABEL_24;
  while ( 2 )
  {
    while ( 2 )
    {
      v21 = *(_DWORD *)(a1 + 264);
      v22 = *v15;
      if ( v21 )
      {
        v17 = v21 - 1;
        v18 = *(_QWORD *)(a1 + 248);
        v19 = (v21 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v20 = *(_QWORD **)(v18 + 8LL * v19);
        if ( v22 != v20 )
        {
          v25 = 1;
          while ( v20 != (_QWORD *)-8LL )
          {
            v19 = v17 & (v25 + v19);
            v20 = *(_QWORD **)(v18 + 8LL * v19);
            if ( v22 == v20 )
              goto LABEL_19;
            ++v25;
          }
          break;
        }
LABEL_19:
        if ( v16 == (_BYTE *)++v15 )
          goto LABEL_23;
        continue;
      }
      break;
    }
    if ( v22[1] )
      goto LABEL_19;
    sub_15F20C0(v22);
    if ( v16 != (_BYTE *)++v15 )
      continue;
    break;
  }
LABEL_23:
  v16 = v28;
LABEL_24:
  if ( v16 != v30 )
    _libc_free((unsigned __int64)v16);
}
