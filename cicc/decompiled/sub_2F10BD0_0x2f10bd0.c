// Function: sub_2F10BD0
// Address: 0x2f10bd0
//
void __fastcall sub_2F10BD0(__int64 a1, __int64 a2, __int64 *a3, unsigned __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r8
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // r9
  __int64 *v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rdi
  char v13; // al
  int v14; // eax
  __int64 v15; // rax
  bool v16; // zf
  char *v17; // [rsp+8h] [rbp-A8h]
  __int64 v18; // [rsp+18h] [rbp-98h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  __int64 *v21; // [rsp+28h] [rbp-88h]
  __int64 v22; // [rsp+30h] [rbp-80h]
  int v23; // [rsp+38h] [rbp-78h]
  char v24; // [rsp+3Ch] [rbp-74h]
  char v25; // [rsp+40h] [rbp-70h] BYREF

  v4 = a1 + 48;
  v5 = *(_QWORD *)(a1 + 56);
  v21 = (__int64 *)&v25;
  v17 = (char *)a3;
  v20 = 0;
  v22 = 8;
  v23 = 0;
  v24 = 1;
  if ( v5 != a1 + 48 )
  {
    while ( 1 )
    {
      if ( *(_WORD *)(v5 + 68) != 68 )
      {
        if ( *(_WORD *)(v5 + 68) )
        {
          v6 = *(_QWORD *)(v5 + 32);
          v7 = v6 + 40LL * (*(_DWORD *)(v5 + 40) & 0xFFFFFF);
          if ( v6 != v7 )
            break;
        }
      }
LABEL_17:
      if ( (*(_BYTE *)v5 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v5 + 44) & 8) != 0 )
          v5 = *(_QWORD *)(v5 + 8);
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        goto LABEL_19;
    }
    v8 = *(_QWORD *)(v5 + 32);
    while ( 1 )
    {
      if ( *(_BYTE *)v8 != 4 )
        goto LABEL_6;
      v9 = *(_QWORD *)(v8 + 24);
      if ( !v24 )
        goto LABEL_28;
      v10 = v21;
      a4 = HIDWORD(v22);
      a3 = &v21[HIDWORD(v22)];
      if ( v21 != a3 )
      {
        while ( v9 != *v10 )
        {
          if ( a3 == ++v10 )
            goto LABEL_12;
        }
        goto LABEL_6;
      }
LABEL_12:
      if ( HIDWORD(v22) < (unsigned int)v22 )
      {
        ++HIDWORD(v22);
        *a3 = v9;
        ++v20;
LABEL_14:
        v11 = *(unsigned int *)(a2 + 8);
        a4 = *(unsigned int *)(a2 + 12);
        if ( v11 + 1 > a4 )
        {
          v19 = v9;
          sub_C8D5F0(a2, (const void *)(a2 + 16), v11 + 1, 8u, v6, v9);
          v11 = *(unsigned int *)(a2 + 8);
          v9 = v19;
        }
        a3 = *(__int64 **)a2;
        v8 += 40;
        *(_QWORD *)(*(_QWORD *)a2 + 8 * v11) = v9;
        ++*(_DWORD *)(a2 + 8);
        if ( v7 == v8 )
          goto LABEL_17;
      }
      else
      {
LABEL_28:
        v18 = *(_QWORD *)(v8 + 24);
        sub_C8CC70((__int64)&v20, v18, (__int64)a3, a4, v6, v9);
        v9 = v18;
        if ( (_BYTE)a3 )
          goto LABEL_14;
LABEL_6:
        v8 += 40;
        if ( v7 == v8 )
          goto LABEL_17;
      }
    }
  }
LABEL_19:
  v12 = sub_2E31A10(a1, 1);
  v13 = 1;
  if ( v4 != v12 )
  {
    v14 = *(_DWORD *)(v12 + 44);
    if ( (v14 & 4) != 0 || (v14 & 8) == 0 )
      v15 = (*(_QWORD *)(*(_QWORD *)(v12 + 16) + 24LL) >> 8) & 1LL;
    else
      LOBYTE(v15) = sub_2E88A90(v12, 256, 1);
    v13 = v15 ^ 1;
  }
  v16 = v24 == 0;
  *v17 = v13;
  if ( v16 )
    _libc_free((unsigned __int64)v21);
}
