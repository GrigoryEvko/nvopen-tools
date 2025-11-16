// Function: sub_983BD0
// Address: 0x983bd0
//
__int64 __fastcall sub_983BD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 src; // [rsp+0h] [rbp-110h] BYREF
  __int64 v12; // [rsp+8h] [rbp-108h]
  __int64 v13; // [rsp+90h] [rbp-80h]
  unsigned int v14; // [rsp+A0h] [rbp-70h]
  __int64 v15; // [rsp+B0h] [rbp-60h]
  __int64 v16; // [rsp+C0h] [rbp-50h]
  __int64 v17; // [rsp+C8h] [rbp-48h]
  __int64 v18; // [rsp+D8h] [rbp-38h]

  if ( !*(_BYTE *)(a2 + 224) )
  {
    sub_982C80((__int64)&src, (_DWORD *)(*(_QWORD *)(a3 + 40) + 232LL));
    if ( *(_BYTE *)(a2 + 224) )
    {
      sub_97F600((_QWORD *)a2, &src);
    }
    else
    {
      sub_97F4E0(a2, (__int64)&src);
      *(_BYTE *)(a2 + 224) = 1;
    }
    if ( v17 )
      j_j___libc_free_0(v17, v18 - v17);
    if ( v15 )
      j_j___libc_free_0(v15, v16 - v15);
    v6 = v14;
    if ( v14 )
    {
      v7 = v13;
      v8 = v13 + 40LL * v14;
      do
      {
        while ( 1 )
        {
          if ( *(_DWORD *)v7 <= 0xFFFFFFFD )
          {
            v9 = *(_QWORD *)(v7 + 8);
            if ( v9 != v7 + 24 )
              break;
          }
          v7 += 40;
          if ( v8 == v7 )
            goto LABEL_15;
        }
        v10 = *(_QWORD *)(v7 + 24);
        v7 += 40;
        j_j___libc_free_0(v9, v10 + 1);
      }
      while ( v8 != v7 );
LABEL_15:
      v6 = v14;
    }
    sub_C7D6A0(v13, 40 * v6, 8);
  }
  src = a3;
  LOBYTE(v12) = 1;
  sub_981090(a1, a2, a3, v12);
  return a1;
}
