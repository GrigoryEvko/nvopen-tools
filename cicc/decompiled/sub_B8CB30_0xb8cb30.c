// Function: sub_B8CB30
// Address: 0xb8cb30
//
__int64 __fastcall sub_B8CB30(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r13
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 *v11; // r8
  __int64 *v12; // r13
  __int64 *v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // rsi
  __int64 v17; // r13
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+18h] [rbp-78h]
  _BYTE *v26; // [rsp+20h] [rbp-70h] BYREF
  __int64 v27; // [rsp+28h] [rbp-68h]
  _BYTE v28[16]; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v29; // [rsp+40h] [rbp-50h] BYREF
  __int64 v30; // [rsp+48h] [rbp-48h]
  _BYTE v31[64]; // [rsp+50h] [rbp-40h] BYREF

  v29 = v31;
  v30 = 0x200000000LL;
  v25 = a2 + 80 * a3;
  if ( a2 != v25 )
  {
    v3 = a2;
    while ( 1 )
    {
      v4 = sub_B8C130(a1, *(_QWORD *)v3, *(_QWORD *)(v3 + 8));
      v5 = HIDWORD(v30);
      v6 = v4;
      v7 = (unsigned int)v30;
      v8 = (unsigned int)v30 + 1LL;
      if ( v8 > HIDWORD(v30) )
      {
        sub_C8D5F0(&v29, v31, v8, 8);
        v7 = (unsigned int)v30;
      }
      *(_QWORD *)&v29[8 * v7] = v6;
      v9 = *(_DWORD *)(v3 + 24);
      LODWORD(v30) = v30 + 1;
      if ( !v9 )
        goto LABEL_3;
      v10 = 1;
      v26 = v28;
      v27 = 0x100000000LL;
      if ( v9 == 1 )
      {
        v11 = *(__int64 **)(v3 + 16);
        v12 = v11 + 1;
        if ( v11 + 1 != v11 )
          goto LABEL_9;
      }
      else
      {
        sub_C8D5F0(&v26, v28, v9, 8);
        v10 = *(unsigned int *)(v3 + 24);
        v11 = *(__int64 **)(v3 + 16);
        v12 = &v11[v10];
        if ( v12 != v11 )
        {
LABEL_9:
          v13 = v11;
          do
          {
            v14 = sub_B8C140((__int64)a1, *v13, v10, v5);
            v15 = (unsigned int)v27;
            if ( (unsigned __int64)(unsigned int)v27 + 1 > HIDWORD(v27) )
            {
              v24 = v14;
              sub_C8D5F0(&v26, v28, (unsigned int)v27 + 1LL, 8);
              v15 = (unsigned int)v27;
              v14 = v24;
            }
            v5 = (__int64)v26;
            ++v13;
            *(_QWORD *)&v26[8 * v15] = v14;
            v10 = (unsigned int)(v27 + 1);
            LODWORD(v27) = v27 + 1;
          }
          while ( v12 != v13 );
          goto LABEL_13;
        }
      }
      v10 = (unsigned int)v27;
LABEL_13:
      v16 = v26;
      v17 = sub_B9C770(*a1, v26, v10, 0, 1);
      v18 = (unsigned int)v30;
      v19 = (unsigned int)v30 + 1LL;
      if ( v19 > HIDWORD(v30) )
      {
        v16 = v31;
        sub_C8D5F0(&v29, v31, v19, 8);
        v18 = (unsigned int)v30;
      }
      *(_QWORD *)&v29[8 * v18] = v17;
      LODWORD(v30) = v30 + 1;
      if ( v26 == v28 )
      {
LABEL_3:
        v3 += 80;
        if ( v25 == v3 )
          goto LABEL_17;
      }
      else
      {
        _libc_free(v26, v16);
        v3 += 80;
        if ( v25 == v3 )
        {
LABEL_17:
          v20 = v29;
          v21 = (unsigned int)v30;
          goto LABEL_18;
        }
      }
    }
  }
  v20 = v31;
  v21 = 0;
LABEL_18:
  v22 = sub_B9C770(*a1, v20, v21, 0, 1);
  if ( v29 != v31 )
    _libc_free(v29, v20);
  return v22;
}
