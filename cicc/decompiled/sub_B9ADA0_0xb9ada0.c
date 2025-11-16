// Function: sub_B9ADA0
// Address: 0xb9ada0
//
void __fastcall sub_B9ADA0(__int64 a1, unsigned int *a2, __int64 a3)
{
  unsigned int *v3; // r15
  unsigned int *v4; // rax
  unsigned int *v5; // r13
  __int64 i; // rax
  int *v7; // rdx
  _BOOL4 v8; // r10d
  __int64 v9; // rax
  unsigned __int64 v10; // rdx
  int *v11; // [rsp-128h] [rbp-128h]
  _BOOL4 v12; // [rsp-11Ch] [rbp-11Ch]
  unsigned int **v13; // [rsp-100h] [rbp-100h] BYREF
  unsigned int *v14; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 v15; // [rsp-F0h] [rbp-F0h]
  _BYTE v16[128]; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v17; // [rsp-68h] [rbp-68h] BYREF
  int v18; // [rsp-60h] [rbp-60h] BYREF
  __int64 v19; // [rsp-58h] [rbp-58h]
  int *v20; // [rsp-50h] [rbp-50h]
  int *v21; // [rsp-48h] [rbp-48h]
  __int64 v22; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 7) & 0x20) != 0 )
  {
    v14 = (unsigned int *)v16;
    v15 = 0x2000000000LL;
    v18 = 0;
    v19 = 0;
    v20 = &v18;
    v21 = &v18;
    v22 = 0;
    sub_B9AC10((__int64)&v14, a2, &a2[a3]);
    LODWORD(v13) = 38;
    if ( v22 )
    {
      sub_B99770((__int64)&v17, (unsigned int *)&v13);
      goto LABEL_8;
    }
    v3 = &v14[(unsigned int)v15];
    if ( v14 == v3 )
    {
      if ( (unsigned int)v15 > 0x1FuLL )
      {
LABEL_26:
        LODWORD(v15) = 0;
        sub_B99770((__int64)&v17, (unsigned int *)&v13);
LABEL_8:
        v13 = &v14;
        sub_B98130(a1, (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_B8E0B0, (__int64)&v13);
        sub_B8FA90(v19);
        if ( v14 != (unsigned int *)v16 )
          _libc_free(v14, sub_B8E0B0);
        return;
      }
    }
    else
    {
      v4 = v14;
      while ( *v4 != 38 )
      {
        if ( v3 == ++v4 )
          goto LABEL_12;
      }
      if ( v3 != v4 )
        goto LABEL_8;
LABEL_12:
      if ( (unsigned int)v15 > 0x1FuLL )
      {
        v5 = v14;
        for ( i = sub_B9AB10(&v17, (__int64)&v18, v14); ; i = sub_B9AB10(&v17, (__int64)&v18, v5) )
        {
          if ( v7 )
          {
            v8 = i || v7 == &v18 || *v5 < v7[8];
            v11 = v7;
            v12 = v8;
            v9 = sub_22077B0(40);
            *(_DWORD *)(v9 + 32) = *v5;
            sub_220F040(v12, v9, v11, &v18);
            ++v22;
          }
          if ( v3 == ++v5 )
            break;
        }
        goto LABEL_26;
      }
    }
    v10 = (unsigned int)v15 + 1LL;
    if ( v10 > HIDWORD(v15) )
    {
      sub_C8D5F0(&v14, v16, v10, 4);
      v3 = &v14[(unsigned int)v15];
    }
    *v3 = 38;
    LODWORD(v15) = v15 + 1;
    goto LABEL_8;
  }
}
