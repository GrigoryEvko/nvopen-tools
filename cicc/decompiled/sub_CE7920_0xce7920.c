// Function: sub_CE7920
// Address: 0xce7920
//
__int64 __fastcall sub_CE7920(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  unsigned int v5; // eax
  __int64 v6; // r9
  unsigned int v7; // r15d
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  __int64 v14; // [rsp+0h] [rbp-D0h]
  _BYTE *v15; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+18h] [rbp-B8h]
  _BYTE v17[176]; // [rsp+20h] [rbp-B0h] BYREF

  v15 = v17;
  v16 = 0x1000000000LL;
  LOBYTE(v5) = sub_CE7690(a1, a2, a3, (__int64)&v15, 0);
  v7 = v5;
  if ( (_BYTE)v5 )
  {
    v9 = (unsigned int)v16;
    if ( (_DWORD)v16 )
    {
      v10 = *(unsigned int *)(a4 + 8);
      v11 = 0;
      a2 = (const void *)(a4 + 16);
      do
      {
        v12 = *(_QWORD *)&v15[8 * v11];
        v13 = *(_QWORD **)(v12 + 24);
        if ( *(_DWORD *)(v12 + 32) > 0x40u )
          v13 = (_QWORD *)*v13;
        if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
        {
          v14 = v9;
          sub_C8D5F0(a4, a2, v10 + 1, 4u, v9, v6);
          v10 = *(unsigned int *)(a4 + 8);
          v9 = v14;
        }
        ++v11;
        *(_DWORD *)(*(_QWORD *)a4 + 4 * v10) = (_DWORD)v13;
        v10 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
        *(_DWORD *)(a4 + 8) = v10;
      }
      while ( v9 != v11 );
    }
  }
  if ( v15 != v17 )
    _libc_free(v15, a2);
  return v7;
}
