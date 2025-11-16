// Function: sub_D61BF0
// Address: 0xd61bf0
//
__int64 __fastcall sub_D61BF0(__int64 a1, __int64 *a2, __int64 a3)
{
  _BYTE *v4; // rax
  __int64 v5; // rcx
  _QWORD *v6; // r12
  __int64 v7; // rsi
  _BYTE *v8; // rbx
  __int64 v9; // rdi
  __int64 v10; // rdi
  unsigned int v12; // [rsp+Ch] [rbp-194h] BYREF
  __int64 v13; // [rsp+10h] [rbp-190h] BYREF
  int v14; // [rsp+18h] [rbp-188h]
  __int64 v15; // [rsp+20h] [rbp-180h]
  int v16; // [rsp+28h] [rbp-178h]
  __int64 v17; // [rsp+30h] [rbp-170h] BYREF
  __int64 v18; // [rsp+38h] [rbp-168h]
  _QWORD *v19; // [rsp+40h] [rbp-160h] BYREF
  unsigned int v20; // [rsp+48h] [rbp-158h]
  _BYTE v21[32]; // [rsp+180h] [rbp-20h] BYREF

  if ( !a2[3] )
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 8) = 1;
    *(_DWORD *)(a1 + 24) = 1;
    return a1;
  }
  v17 = 0;
  v4 = &v19;
  v18 = 1;
  do
  {
    *(_QWORD *)v4 = -4096;
    v4 += 40;
  }
  while ( v4 != v21 );
  v5 = *(_QWORD *)(a3 + 40);
  v12 = 0;
  sub_D60CE0((__int64)&v13, a2, a3, v5, (_QWORD *)(a3 + 24), 0, (__int64)&v17, &v12);
  *(_DWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v13;
  *(_DWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 16) = v15;
  if ( (v18 & 1) != 0 )
  {
    v8 = v21;
    v6 = &v19;
    do
    {
LABEL_7:
      if ( *v6 != -4096 && *v6 != -8192 )
      {
        if ( *((_DWORD *)v6 + 8) > 0x40u )
        {
          v9 = v6[3];
          if ( v9 )
            j_j___libc_free_0_0(v9);
        }
        if ( *((_DWORD *)v6 + 4) > 0x40u )
        {
          v10 = v6[1];
          if ( v10 )
            j_j___libc_free_0_0(v10);
        }
      }
      v6 += 5;
    }
    while ( v6 != (_QWORD *)v8 );
    if ( (v18 & 1) != 0 )
      return a1;
    v6 = v19;
    v7 = 5LL * v20;
    goto LABEL_20;
  }
  v6 = v19;
  v7 = 5LL * v20;
  if ( v20 )
  {
    v8 = &v19[v7];
    if ( v19 != &v19[v7] )
      goto LABEL_7;
  }
LABEL_20:
  sub_C7D6A0((__int64)v6, v7 * 8, 8);
  return a1;
}
