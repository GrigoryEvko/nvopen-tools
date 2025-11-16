// Function: sub_29ABAB0
// Address: 0x29abab0
//
void __fastcall sub_29ABAB0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  unsigned __int64 v4; // rax
  __int64 v5; // r12
  int v6; // ebx
  unsigned int v7; // r14d
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 v14; // r8
  int v15; // eax
  char *v16; // rax
  __int64 v17; // rax
  char v18; // dl
  int v19; // edi
  const void *v20; // [rsp+8h] [rbp-78h]
  __int64 v21; // [rsp+10h] [rbp-70h]
  __int64 v22; // [rsp+18h] [rbp-68h]
  __int64 v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  char *v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h]
  int v27; // [rsp+38h] [rbp-48h]
  char v28; // [rsp+3Ch] [rbp-44h]
  char v29; // [rsp+40h] [rbp-40h] BYREF

  v25 = &v29;
  v1 = *(_QWORD *)(a1 + 88);
  v2 = *(unsigned int *)(a1 + 96);
  *(_DWORD *)(a1 + 112) = 0;
  v24 = 0;
  v26 = 2;
  v27 = 0;
  v28 = 1;
  v21 = v1 + 8 * v2;
  v20 = (const void *)(a1 + 120);
  if ( v21 != v1 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)v1 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v4 != *(_QWORD *)v1 + 48LL )
      {
        if ( !v4 )
          BUG();
        v5 = v4 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 <= 0xA )
        {
          v6 = sub_B46E30(v5);
          if ( v6 )
            break;
        }
      }
LABEL_19:
      v1 += 8;
      if ( v21 == v1 )
      {
        if ( !v28 )
          _libc_free((unsigned __int64)v25);
        return;
      }
    }
    v7 = 0;
    while ( 1 )
    {
      v11 = sub_B46EC0(v5, v7);
      v13 = *(_QWORD *)(a1 + 64);
      v14 = v11;
      v15 = *(_DWORD *)(a1 + 80);
      if ( v15 )
      {
        v8 = (unsigned int)(v15 - 1);
        v9 = v8 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v10 = *(_QWORD *)(v13 + 8LL * v9);
        if ( v14 == v10 )
          goto LABEL_8;
        v19 = 1;
        while ( v10 != -4096 )
        {
          v9 = v8 & (v19 + v9);
          v10 = *(_QWORD *)(v13 + 8LL * v9);
          if ( v14 == v10 )
            goto LABEL_8;
          ++v19;
        }
      }
      if ( !v28 )
        goto LABEL_23;
      v16 = v25;
      v13 = HIDWORD(v26);
      v8 = (__int64)&v25[8 * HIDWORD(v26)];
      if ( v25 != (char *)v8 )
      {
        while ( v14 != *(_QWORD *)v16 )
        {
          v16 += 8;
          if ( (char *)v8 == v16 )
            goto LABEL_14;
        }
        goto LABEL_8;
      }
LABEL_14:
      if ( HIDWORD(v26) < (unsigned int)v26 )
      {
        ++HIDWORD(v26);
        *(_QWORD *)v8 = v14;
        ++v24;
LABEL_16:
        v17 = *(unsigned int *)(a1 + 112);
        if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
        {
          v23 = v14;
          sub_C8D5F0(a1 + 104, v20, v17 + 1, 8u, v14, v12);
          v17 = *(unsigned int *)(a1 + 112);
          v14 = v23;
        }
        ++v7;
        *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8 * v17) = v14;
        ++*(_DWORD *)(a1 + 112);
        if ( v6 == v7 )
          goto LABEL_19;
      }
      else
      {
LABEL_23:
        v22 = v14;
        sub_C8CC70((__int64)&v24, v14, v8, v13, v14, v12);
        v14 = v22;
        if ( v18 )
          goto LABEL_16;
LABEL_8:
        if ( v6 == ++v7 )
          goto LABEL_19;
      }
    }
  }
}
