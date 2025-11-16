// Function: sub_9B92A0
// Address: 0x9b92a0
//
__int64 __fastcall sub_9B92A0(__int64 a1, int a2, unsigned int *a3)
{
  unsigned int v3; // eax
  int v6; // r12d
  unsigned int v7; // r13d
  unsigned int v8; // edx
  unsigned int v9; // esi
  int *v10; // rax
  int v11; // r10d
  _BOOL8 v12; // r14
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int v16; // edx
  __int64 v17; // rdi
  unsigned int v18; // ecx
  int v19; // eax
  int v20; // r9d
  _BYTE *v21; // rdi
  __int64 v22; // rsi
  __int64 v23; // r12
  __int64 v26; // [rsp+18h] [rbp-C8h]
  _BYTE *v27; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-B8h]
  _BYTE v29[176]; // [rsp+30h] [rbp-B0h] BYREF

  v3 = *a3;
  if ( a3[6] == *a3 )
    return 0;
  v27 = v29;
  v28 = 0x1000000000LL;
  if ( a2 )
  {
    v6 = 0;
    while ( 1 )
    {
      v7 = 0;
      if ( v3 )
        break;
LABEL_13:
      if ( a2 == ++v6 )
      {
        v21 = v27;
        v22 = (unsigned int)v28;
        goto LABEL_20;
      }
      v3 = *a3;
    }
    while ( 1 )
    {
      v16 = a3[8];
      v17 = *((_QWORD *)a3 + 2);
      v18 = v7 + a3[10];
      if ( !v16 )
        goto LABEL_12;
      v8 = v16 - 1;
      v9 = v8 & (37 * v18);
      v10 = (int *)(v17 + 16LL * v9);
      v11 = *v10;
      if ( v18 != *v10 )
        break;
LABEL_7:
      v12 = *((_QWORD *)v10 + 1) != 0;
LABEL_8:
      v13 = sub_BCB2A0(*(_QWORD *)(a1 + 72));
      v14 = sub_ACD640(v13, v12, 0);
      v15 = (unsigned int)v28;
      if ( (unsigned __int64)(unsigned int)v28 + 1 > HIDWORD(v28) )
      {
        v26 = v14;
        sub_C8D5F0(&v27, v29, (unsigned int)v28 + 1LL, 8);
        v15 = (unsigned int)v28;
        v14 = v26;
      }
      ++v7;
      *(_QWORD *)&v27[8 * v15] = v14;
      LODWORD(v28) = v28 + 1;
      if ( *a3 <= v7 )
        goto LABEL_13;
    }
    v19 = 1;
    while ( v11 != 0x7FFFFFFF )
    {
      v20 = v19 + 1;
      v9 = v8 & (v19 + v9);
      v10 = (int *)(v17 + 16LL * v9);
      v11 = *v10;
      if ( v18 == *v10 )
        goto LABEL_7;
      v19 = v20;
    }
LABEL_12:
    v12 = 0;
    goto LABEL_8;
  }
  v21 = v29;
  v22 = 0;
LABEL_20:
  v23 = sub_AD3730(v21, v22);
  if ( v27 != v29 )
    _libc_free(v27, v22);
  return v23;
}
