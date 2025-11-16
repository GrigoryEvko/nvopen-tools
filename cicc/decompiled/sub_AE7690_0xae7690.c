// Function: sub_AE7690
// Address: 0xae7690
//
unsigned __int64 *__fastcall sub_AE7690(unsigned __int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  char *v6; // rdi
  char *v7; // r8
  char *v8; // rbx
  unsigned __int64 v9; // r15
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // r10
  __int64 v14; // rax
  unsigned __int64 v15; // r11
  __int64 v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // [rsp+10h] [rbp-80h]
  char *v19; // [rsp+18h] [rbp-78h]
  char *v20; // [rsp+18h] [rbp-78h]
  char *v21; // [rsp+20h] [rbp-70h] BYREF
  int v22; // [rsp+28h] [rbp-68h]
  char v23; // [rsp+30h] [rbp-60h] BYREF

  if ( (*(_BYTE *)(a2 + 7) & 8) != 0 && (v4 = sub_B91390(a2)) != 0 )
  {
    v5 = v4 + 8;
    sub_B967C0(&v21, v4 + 8);
    v6 = v21;
    v7 = &v21[8 * v22];
    if ( v7 != v21 )
    {
      v8 = v21;
      v9 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v10 = *(_QWORD *)v8;
          if ( !*(_BYTE *)(*(_QWORD *)v8 + 64LL) )
            break;
LABEL_7:
          v8 += 8;
          if ( v7 == v8 )
            goto LABEL_15;
        }
        if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        {
          v9 = v10 & 0xFFFFFFFFFFFFFFFBLL;
          goto LABEL_7;
        }
        if ( (v9 & 4) != 0 )
        {
          v11 = *(unsigned int *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 8);
          v12 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          v13 = v9 & 0xFFFFFFFFFFFFFFF8LL;
        }
        else
        {
          v19 = v7;
          v14 = sub_22077B0(48);
          v7 = v19;
          v15 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v14 )
          {
            *(_QWORD *)v14 = v14 + 16;
            *(_QWORD *)(v14 + 8) = 0x400000000LL;
          }
          v16 = v14;
          v12 = v14 & 0xFFFFFFFFFFFFFFF8LL;
          v17 = *(unsigned int *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 8);
          v9 = v16 | 4;
          v13 = v12;
          if ( v17 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
          {
            v5 = v12 + 16;
            v18 = v15;
            sub_C8D5F0(v12, v12 + 16, v17 + 1, 8);
            v17 = *(unsigned int *)(v12 + 8);
            v13 = v12;
            v15 = v18;
            v7 = v19;
          }
          *(_QWORD *)(*(_QWORD *)v12 + 8 * v17) = v15;
          v11 = (unsigned int)(*(_DWORD *)(v12 + 8) + 1);
          *(_DWORD *)(v12 + 8) = v11;
        }
        if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
        {
          v5 = v12 + 16;
          v20 = v7;
          sub_C8D5F0(v13, v12 + 16, v11 + 1, 8);
          v11 = *(unsigned int *)(v12 + 8);
          v7 = v20;
        }
        v8 += 8;
        *(_QWORD *)(*(_QWORD *)v12 + 8 * v11) = v10;
        ++*(_DWORD *)(v12 + 8);
        if ( v7 == v8 )
        {
LABEL_15:
          v6 = v21;
          goto LABEL_16;
        }
      }
    }
    v9 = 0;
LABEL_16:
    if ( v6 != &v23 )
      _libc_free(v6, v5);
    *a1 = v9;
  }
  else
  {
    *a1 = 0;
  }
  return a1;
}
