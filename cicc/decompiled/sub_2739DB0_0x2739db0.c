// Function: sub_2739DB0
// Address: 0x2739db0
//
void __fastcall sub_2739DB0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned int v8; // r15d
  __int64 v9; // rax
  char *v10; // r8
  char *v11; // rdx
  _BYTE *v12; // rdi
  size_t v13; // rdx
  _BYTE *v14; // rax
  __int64 v15; // [rsp+0h] [rbp-90h]
  __int64 v16; // [rsp+0h] [rbp-90h]
  _BYTE *v17; // [rsp+8h] [rbp-88h] BYREF
  __int64 v18; // [rsp+10h] [rbp-80h]
  _BYTE dest[120]; // [rsp+18h] [rbp-78h] BYREF

  v7 = *a2;
  v8 = *((_DWORD *)a2 + 4);
  v18 = 0x300000000LL;
  v17 = dest;
  v15 = v7;
  if ( v8 )
  {
    if ( v8 > 3 )
    {
      sub_C8D5F0((__int64)&v17, dest, v8, 0x18u, a5, a6);
      v12 = v17;
      v13 = 24LL * *((unsigned int *)a2 + 4);
      if ( !v13 )
      {
LABEL_9:
        v16 = -v15;
        LODWORD(v18) = v8;
        v14 = &v12[24 * v8];
        do
        {
          *(_QWORD *)v12 = -*(_QWORD *)v12;
          v12 += 24;
        }
        while ( v14 != v12 );
        v11 = v17;
        v9 = v16;
        v10 = &v17[24 * (unsigned int)v18];
        goto LABEL_3;
      }
    }
    else
    {
      v12 = dest;
      v13 = 24LL * v8;
    }
    memcpy(v12, (const void *)a2[1], v13);
    v12 = v17;
    goto LABEL_9;
  }
  v9 = -v7;
  v10 = dest;
  v11 = dest;
LABEL_3:
  *(_QWORD *)a1 += v9;
  sub_2739AD0(a1 + 8, (__m128i *)(*(_QWORD *)(a1 + 8) + 24LL * *(unsigned int *)(a1 + 16)), v11, v10);
  if ( v17 != dest )
    _libc_free((unsigned __int64)v17);
}
