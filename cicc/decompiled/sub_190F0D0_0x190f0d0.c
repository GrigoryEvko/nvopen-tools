// Function: sub_190F0D0
// Address: 0x190f0d0
//
__int64 __fastcall sub_190F0D0(__int64 a1, __int64 a2, __int64 *a3)
{
  int v4; // r14d
  __int64 v6; // r15
  __int64 *v7; // rdi
  int v8; // r14d
  __int64 v9; // rax
  int v10; // eax
  unsigned int v11; // r8d
  int v12; // r11d
  __int64 v13; // r10
  unsigned int i; // ecx
  __int64 v15; // r13
  int v16; // r9d
  bool v17; // al
  __int64 v18; // rdx
  size_t v19; // rdx
  int v20; // eax
  int v21; // [rsp+0h] [rbp-E0h]
  unsigned int v22; // [rsp+4h] [rbp-DCh]
  unsigned int v23; // [rsp+4h] [rbp-DCh]
  int v24; // [rsp+8h] [rbp-D8h]
  int v25; // [rsp+8h] [rbp-D8h]
  unsigned int v26; // [rsp+Ch] [rbp-D4h]
  unsigned int v27; // [rsp+Ch] [rbp-D4h]
  __int64 v28; // [rsp+10h] [rbp-D0h]
  __int64 v29; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v30; // [rsp+28h] [rbp-B8h] BYREF
  int v31; // [rsp+30h] [rbp-B0h]
  char v32; // [rsp+40h] [rbp-A0h]
  char *v33; // [rsp+48h] [rbp-98h]
  __int64 v34; // [rsp+50h] [rbp-90h]
  char v35; // [rsp+58h] [rbp-88h] BYREF
  int v36[4]; // [rsp+70h] [rbp-70h] BYREF
  char v37; // [rsp+80h] [rbp-60h]
  char *v38; // [rsp+88h] [rbp-58h]
  __int64 v39; // [rsp+90h] [rbp-50h]
  char v40; // [rsp+98h] [rbp-48h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    v7 = *(__int64 **)(a2 + 24);
    v38 = &v40;
    v8 = v4 - 1;
    v33 = &v35;
    v34 = 0x400000000LL;
    v39 = 0x400000000LL;
    v9 = *(unsigned int *)(a2 + 32);
    v31 = -1;
    v32 = 0;
    v36[0] = -2;
    v37 = 0;
    v30 = sub_1597510(v7, (__int64)v7 + 4 * v9);
    v10 = sub_190BFA0((int *)a2, (__int64 *)(a2 + 8), (__int64 *)&v30);
    v11 = *(_DWORD *)a2;
    v12 = 1;
    v13 = 0;
    for ( i = v8 & v10; ; i = v8 & (v24 + v26) )
    {
      v15 = v6 + ((unsigned __int64)i << 6);
      v16 = *(_DWORD *)v15;
      if ( v11 == *(_DWORD *)v15 )
      {
        if ( v11 > 0xFFFFFFFD )
          goto LABEL_15;
        if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(v15 + 8) )
        {
          v18 = *(unsigned int *)(a2 + 32);
          if ( v18 == *(_DWORD *)(v15 + 32) )
          {
            v19 = 4 * v18;
            v21 = *(_DWORD *)v15;
            v23 = v11;
            v25 = v12;
            v27 = i;
            v29 = v13;
            if ( !v19
              || (v20 = memcmp(*(const void **)(a2 + 24), *(const void **)(v15 + 24), v19),
                  v13 = v29,
                  i = v27,
                  v12 = v25,
                  v11 = v23,
                  v16 = v21,
                  !v20) )
            {
LABEL_15:
              *a3 = v15;
              return 1;
            }
          }
        }
      }
      if ( v16 == -1 )
        break;
      v22 = v11;
      v24 = v12;
      v26 = i;
      v28 = v13;
      v17 = sub_190A670(v15, (__int64)v36);
      v13 = v28;
      v11 = v22;
      if ( !v28 && v17 )
        v13 = v15;
      v12 = v24 + 1;
    }
    if ( !v13 )
      v13 = v15;
    *a3 = v13;
    return 0;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
