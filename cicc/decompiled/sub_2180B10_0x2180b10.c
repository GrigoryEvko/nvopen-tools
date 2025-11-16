// Function: sub_2180B10
// Address: 0x2180b10
//
__int64 __fastcall sub_2180B10(__int64 a1)
{
  __int64 v2; // r13
  _DWORD *v3; // rax
  unsigned int v4; // r12d
  __int64 v5; // rbx
  __int64 v6; // r13
  _DWORD *v7; // rdx
  unsigned int v8; // r15d
  __int64 *v9; // rcx
  unsigned int v10; // eax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // rdi
  int v15; // esi
  unsigned int v16; // ecx
  __int64 *v17; // [rsp+8h] [rbp-78h]
  __int64 v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  __int64 v20; // [rsp+20h] [rbp-60h]
  int v21; // [rsp+28h] [rbp-58h]
  __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+38h] [rbp-48h]
  __int64 v24; // [rsp+40h] [rbp-40h]
  int v25; // [rsp+48h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 64);
  v3 = *(_DWORD **)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 16);
  v5 = *(_QWORD *)(v2 + 32);
  v6 = v2 + 24;
  v7 = &v3[*(unsigned int *)(a1 + 24)];
  if ( v4 )
  {
    if ( v3 == v7 )
      goto LABEL_20;
    while ( *v3 > 0xFFFFFFFD )
    {
      if ( ++v3 == v7 )
        goto LABEL_20;
    }
    if ( v3 == v7 )
    {
LABEL_20:
      v4 = 0;
    }
    else
    {
      v12 = *(_QWORD *)(a1 + 72);
      v4 = 0;
      v13 = *(_QWORD *)(*(_QWORD *)(a1 + 80) + 24LL);
      v14 = *(_QWORD *)(v12 + 280);
      v15 = *(_DWORD *)(v12 + 288) * ((__int64)(*(_QWORD *)(v12 + 264) - *(_QWORD *)(v12 + 256)) >> 3);
LABEL_23:
      v16 = v4 + 2;
      ++v4;
      if ( *(_DWORD *)(v14
                     + 24LL
                     * (v15
                      + (unsigned int)*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(v13 + 16LL * (*v3 & 0x7FFFFFFF))
                                                                      & 0xFFFFFFFFFFFFFFF8LL)
                                                          + 24LL))) > 0x20u )
        v4 = v16;
      while ( ++v3 != v7 )
      {
        if ( *v3 <= 0xFFFFFFFD )
        {
          if ( v3 != v7 )
            goto LABEL_23;
          break;
        }
      }
    }
  }
  if ( v5 != v6 )
  {
    v18 = 0;
    v8 = 0;
    v9 = &v22;
    v19 = 0;
    v20 = 0;
    v21 = 0;
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    do
    {
      while ( 1 )
      {
        if ( **(_WORD **)(v5 + 16) )
        {
          if ( **(_WORD **)(v5 + 16) != 45 )
          {
            if ( *(_DWORD *)(v5 + 40) )
            {
              v17 = v9;
              v10 = sub_2180460(a1, v5, (__int64)&v18, (__int64)v9);
              v9 = v17;
              if ( v8 < v10 )
                v8 = v10;
            }
          }
        }
        if ( (*(_BYTE *)v5 & 4) == 0 )
          break;
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 == v6 )
          goto LABEL_12;
      }
      while ( (*(_BYTE *)(v5 + 46) & 8) != 0 )
        v5 = *(_QWORD *)(v5 + 8);
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v5 != v6 );
LABEL_12:
    v4 += v8;
    j___libc_free_0(v23);
    j___libc_free_0(v19);
  }
  return v4;
}
