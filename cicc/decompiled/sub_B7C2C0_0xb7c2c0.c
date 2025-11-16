// Function: sub_B7C2C0
// Address: 0xb7c2c0
//
__int64 __fastcall sub_B7C2C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 v3; // rdx
  int v4; // eax
  _BYTE *v5; // rdi
  __int64 *v6; // r12
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r8
  unsigned int v10; // r9d
  __int64 v12; // r15
  __int64 *v13; // rax
  __int64 *v14; // r15
  __int64 *i; // r13
  __int64 v16; // [rsp+18h] [rbp-88h] BYREF
  __int64 v17; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h]
  __int64 v19; // [rsp+30h] [rbp-70h]
  __int64 v20; // [rsp+38h] [rbp-68h]
  _BYTE *v21; // [rsp+40h] [rbp-60h]
  __int64 v22; // [rsp+48h] [rbp-58h]
  _BYTE v23[80]; // [rsp+50h] [rbp-50h] BYREF

  v22 = 0x400000000LL;
  v2 = *(__int64 **)(a1 + 1752);
  v3 = *(unsigned int *)(a1 + 1768);
  v4 = *(_DWORD *)(a1 + 1760);
  v17 = 0;
  v5 = v23;
  v18 = 0;
  v19 = 0;
  v6 = &v2[v3];
  v20 = 0;
  v21 = v23;
  if ( v4 )
  {
    if ( v6 == v2 )
    {
      v4 = 0;
    }
    else
    {
      while ( *v2 == -4096 || *v2 == -8192 )
      {
        if ( ++v2 == v6 )
          goto LABEL_28;
      }
      if ( v2 == v6 )
      {
LABEL_28:
        v5 = v23;
        v4 = 0;
        goto LABEL_6;
      }
LABEL_30:
      v16 = *v2;
      if ( !*(_QWORD *)(v16 + 16) )
      {
        a2 = (__int64)&v16;
        sub_B7BC80((__int64)&v17, &v16);
      }
      while ( ++v2 != v6 )
      {
        if ( *v2 != -8192 && *v2 != -4096 )
        {
          if ( v2 != v6 )
            goto LABEL_30;
          break;
        }
      }
      v4 = v22;
      v5 = v21;
    }
  }
LABEL_6:
  while ( v4 )
  {
    v7 = *(_QWORD *)&v5[8 * v4 - 8];
    v8 = (unsigned int)v20;
    if ( (_DWORD)v20 )
    {
      v8 = ((_DWORD)v20 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      a2 = v18 + 8 * v8;
      v9 = *(_QWORD *)a2;
      if ( v7 == *(_QWORD *)a2 )
      {
LABEL_3:
        *(_QWORD *)a2 = -8192;
        v4 = v22;
        LODWORD(v19) = v19 - 1;
        ++HIDWORD(v19);
      }
      else
      {
        a2 = 1;
        while ( v9 != -4096 )
        {
          v10 = a2 + 1;
          v8 = ((_DWORD)v20 - 1) & (unsigned int)(a2 + v8);
          a2 = v18 + 8LL * (unsigned int)v8;
          v9 = *(_QWORD *)a2;
          if ( v7 == *(_QWORD *)a2 )
            goto LABEL_3;
          a2 = v10;
        }
      }
    }
    LODWORD(v22) = --v4;
    if ( !*(_QWORD *)(v7 + 16) )
    {
      v12 = 4LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
      if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
      {
        v13 = *(__int64 **)(v7 - 8);
        v14 = &v13[v12];
      }
      else
      {
        v13 = (__int64 *)(v7 - v12 * 8);
        v14 = (__int64 *)v7;
      }
      for ( i = v13; v14 != i; i += 4 )
      {
        v8 = *i;
        if ( *(_BYTE *)*i == 9 )
        {
          a2 = (__int64)&v16;
          v16 = *i;
          sub_B7BC80((__int64)&v17, &v16);
        }
      }
      sub_ACFDF0((__int64 *)v7, a2, v8);
      v4 = v22;
    }
    v5 = v21;
  }
  if ( v5 != v23 )
    _libc_free(v5, a2);
  return sub_C7D6A0(v18, 8LL * (unsigned int)v20, 8);
}
