// Function: sub_D5C200
// Address: 0xd5c200
//
__int64 __fastcall sub_D5C200(__int64 a1, char *a2, unsigned int a3, int a4)
{
  __int64 v4; // r14
  char v7; // dl
  unsigned int v8; // edx
  unsigned int v10; // r12d
  __int64 v11; // rcx
  __int64 v12; // r12
  char v13; // al
  unsigned int v14; // r12d
  unsigned int v15; // eax
  unsigned int v16; // edx
  unsigned int v17; // eax
  __int64 v18; // [rsp+10h] [rbp-F0h]
  unsigned int v19; // [rsp+24h] [rbp-DCh]
  const void *v20; // [rsp+30h] [rbp-D0h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-C8h]
  char v22; // [rsp+40h] [rbp-C0h]
  const void *v23; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v24; // [rsp+58h] [rbp-A8h]
  char v25; // [rsp+60h] [rbp-A0h]
  const void *v26; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v27; // [rsp+78h] [rbp-88h]
  char v28; // [rsp+80h] [rbp-80h]
  const void *v29; // [rsp+90h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+98h] [rbp-68h]
  char v31; // [rsp+A0h] [rbp-60h]
  const void *v32; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v33; // [rsp+B8h] [rbp-48h]
  char v34; // [rsp+C0h] [rbp-40h]

  v4 = a1;
  if ( a4 == 4 )
    goto LABEL_8;
  v7 = *a2;
  if ( *a2 == 17 )
  {
    v8 = *((_DWORD *)a2 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
      sub_C43780(a1, (const void **)a2 + 3);
    else
      *(_QWORD *)a1 = *((_QWORD *)a2 + 3);
    goto LABEL_5;
  }
  if ( v7 != 86 )
  {
    if ( v7 == 84 )
    {
      v10 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
      if ( v10 )
      {
        v11 = (unsigned int)(a4 + 1);
        v19 = v11;
        sub_D5C200(&v20, **((_QWORD **)a2 - 1), a3, v11);
        if ( v22 )
        {
          if ( v10 == 1 )
          {
LABEL_29:
            *(_DWORD *)(v4 + 8) = v21;
            *(_QWORD *)v4 = v20;
LABEL_5:
            *(_BYTE *)(v4 + 16) = 1;
            return v4;
          }
          v18 = 32LL * v10;
          v12 = 32;
          while ( 1 )
          {
            sub_D5C200(&v23, *(_QWORD *)(*((_QWORD *)a2 - 1) + v12), a3, v19);
            v31 = 0;
            if ( v25 )
            {
              v30 = v24;
              if ( v24 > 0x40 )
                sub_C43780((__int64)&v29, &v23);
              else
                v29 = v23;
              v31 = 1;
            }
            v28 = 0;
            if ( v22 )
            {
              v27 = v21;
              if ( v21 > 0x40 )
                sub_C43780((__int64)&v26, &v20);
              else
                v26 = v20;
              v28 = 1;
            }
            sub_D5C150((__int64)&v32, (__int64)&v26, (__int64)&v29, a3);
            if ( v22 )
            {
              if ( !v34 )
              {
                v22 = 0;
                if ( v21 <= 0x40 || !v20 )
                  goto LABEL_23;
                j_j___libc_free_0_0(v20);
                v13 = v34;
                goto LABEL_22;
              }
              if ( v21 > 0x40 && v20 )
              {
                j_j___libc_free_0_0(v20);
                v13 = v34;
                v20 = v32;
                v16 = v33;
                v33 = 0;
                v21 = v16;
LABEL_22:
                if ( !v13 )
                  goto LABEL_23;
                goto LABEL_37;
              }
              v20 = v32;
              v17 = v33;
              v33 = 0;
              v21 = v17;
            }
            else
            {
              if ( !v34 )
                goto LABEL_23;
              v15 = v33;
              v22 = 1;
              v33 = 0;
              v21 = v15;
              v20 = v32;
            }
LABEL_37:
            v34 = 0;
            if ( v33 > 0x40 && v32 )
              j_j___libc_free_0_0(v32);
LABEL_23:
            if ( v28 )
            {
              v28 = 0;
              if ( v27 > 0x40 )
              {
                if ( v26 )
                  j_j___libc_free_0_0(v26);
              }
            }
            if ( v31 )
            {
              v31 = 0;
              if ( v30 > 0x40 )
              {
                if ( v29 )
                  j_j___libc_free_0_0(v29);
              }
            }
            if ( v25 )
            {
              v25 = 0;
              if ( v24 > 0x40 )
              {
                if ( v23 )
                  j_j___libc_free_0_0(v23);
              }
            }
            if ( !v22 )
            {
              v4 = a1;
              break;
            }
            v12 += 32;
            if ( v18 == v12 )
            {
              v4 = a1;
              goto LABEL_29;
            }
          }
        }
      }
    }
LABEL_8:
    *(_BYTE *)(v4 + 16) = 0;
    return v4;
  }
  v14 = a4 + 1;
  sub_D5C200(&v32, *((_QWORD *)a2 - 4), a3, (unsigned int)(a4 + 1));
  sub_D5C200(&v29, *((_QWORD *)a2 - 8), a3, v14);
  sub_D5C150(a1, (__int64)&v29, (__int64)&v32, a3);
  if ( v31 )
  {
    v31 = 0;
    if ( v30 > 0x40 )
    {
      if ( v29 )
        j_j___libc_free_0_0(v29);
    }
  }
  if ( v34 )
  {
    v34 = 0;
    if ( v33 > 0x40 )
    {
      if ( v32 )
        j_j___libc_free_0_0(v32);
    }
  }
  return v4;
}
