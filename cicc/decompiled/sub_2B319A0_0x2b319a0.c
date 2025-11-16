// Function: sub_2B319A0
// Address: 0x2b319a0
//
void __fastcall sub_2B319A0(__int64 a1, int *a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r8
  __int64 v4; // r9
  __int64 v6; // rbx
  unsigned __int64 v7; // rcx
  int v8; // edi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  size_t v12; // [rsp-88h] [rbp-88h]
  __int64 v14; // [rsp-80h] [rbp-80h]
  _BYTE *v16; // [rsp-78h] [rbp-78h] BYREF
  __int64 v17; // [rsp-70h] [rbp-70h]
  _BYTE v18[104]; // [rsp-68h] [rbp-68h] BYREF

  if ( a3 )
  {
    v3 = a3;
    v4 = 4 * a3;
    v6 = *(unsigned int *)(a1 + 8);
    if ( (_DWORD)v6 )
    {
      v16 = v18;
      v17 = 0xC00000000LL;
      if ( a3 > 0xC )
      {
        v12 = 4 * a3;
        sub_C8D5F0((__int64)&v16, v18, a3, 4u, a3, v4);
        v4 = v12;
        v3 = a3;
        if ( v12 )
        {
          memset(v16, 255, v12);
          v3 = a3;
        }
        LODWORD(v17) = v3;
        v8 = v3;
        if ( (int)v3 <= 0 )
          goto LABEL_14;
        v7 = *(unsigned int *)(a1 + 8);
        if ( v7 > v3 )
          v7 = v3;
      }
      else
      {
        if ( v4 )
        {
          memset(v18, 255, 4 * a3);
          v3 = a3;
        }
        v7 = (unsigned int)v6;
        LODWORD(v17) = v3;
        v8 = v3;
        if ( (unsigned int)v6 > v3 )
          v7 = v3;
      }
      v9 = 0;
      do
      {
        v10 = a2[v9];
        if ( (_DWORD)v10 != -1 && (int)v10 < (int)v7 )
        {
          v3 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(*(_QWORD *)a1 + 4 * v10);
          if ( (int)v10 < (int)v7 )
          {
            v3 = (unsigned __int64)v16;
            *(_DWORD *)&v16[4 * v9] = v10;
          }
        }
        ++v9;
      }
      while ( v8 > (int)v9 );
LABEL_14:
      sub_2B310D0(a1, (__int64)&v16, v10, v7, v3, v4);
      if ( v16 != v18 )
        _libc_free((unsigned __int64)v16);
      return;
    }
    v11 = v4 >> 2;
    if ( v4 >> 2 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      v14 = 4 * a3;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v4 >> 2, 4u, a3, v4);
      v6 = *(unsigned int *)(a1 + 8);
      v4 = v14;
    }
    if ( v4 )
    {
      memcpy((void *)(*(_QWORD *)a1 + 4 * v6), a2, v4);
      LODWORD(v6) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v11 + v6;
  }
}
