// Function: sub_97F600
// Address: 0x97f600
//
void *__fastcall sub_97F600(_QWORD *dest, _QWORD *src)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v3 = *((unsigned int *)dest + 40);
  if ( (_DWORD)v3 )
  {
    v4 = dest[18];
    v5 = v4 + 40 * v3;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v4 <= 0xFFFFFFFD )
        {
          v6 = *(_QWORD *)(v4 + 8);
          if ( v6 != v4 + 24 )
            break;
        }
        v4 += 40;
        if ( v5 == v4 )
          goto LABEL_7;
      }
      v7 = *(_QWORD *)(v4 + 24);
      v4 += 40;
      j_j___libc_free_0(v6, v7 + 1);
    }
    while ( v5 != v4 );
LABEL_7:
    v3 = *((unsigned int *)dest + 40);
  }
  sub_C7D6A0(dest[18], 40 * v3, 8);
  dest[19] = 0;
  dest[18] = 0;
  *((_DWORD *)dest + 40) = 0;
  ++dest[17];
  v8 = src[18];
  ++src[17];
  v9 = dest[18];
  dest[18] = v8;
  LODWORD(v8) = *((_DWORD *)src + 38);
  src[18] = v9;
  LODWORD(v9) = *((_DWORD *)dest + 38);
  *((_DWORD *)dest + 38) = v8;
  LODWORD(v8) = *((_DWORD *)src + 39);
  *((_DWORD *)src + 38) = v9;
  LODWORD(v9) = *((_DWORD *)dest + 39);
  *((_DWORD *)dest + 39) = v8;
  LODWORD(v8) = *((_DWORD *)src + 40);
  *((_DWORD *)src + 39) = v9;
  LODWORD(v9) = *((_DWORD *)dest + 40);
  *((_DWORD *)dest + 40) = v8;
  *((_DWORD *)src + 40) = v9;
  *((_BYTE *)dest + 168) = *((_BYTE *)src + 168);
  *((_BYTE *)dest + 169) = *((_BYTE *)src + 169);
  *((_BYTE *)dest + 170) = *((_BYTE *)src + 170);
  *((_BYTE *)dest + 171) = *((_BYTE *)src + 171);
  *((_DWORD *)dest + 43) = *((_DWORD *)src + 43);
  return memmove(dest, src, 0x83u);
}
