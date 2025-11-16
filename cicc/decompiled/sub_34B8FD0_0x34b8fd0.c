// Function: sub_34B8FD0
// Address: 0x34b8fd0
//
void __fastcall sub_34B8FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v8; // r8
  __int64 v9; // r9
  _BYTE *v10; // r12
  _BYTE *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r14
  __int128 v14; // [rsp-10h] [rbp-B0h]
  __int128 v15; // [rsp-10h] [rbp-B0h]
  __int64 v16; // [rsp+18h] [rbp-88h]
  _BYTE *v17; // [rsp+20h] [rbp-80h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h]
  _BYTE v19[112]; // [rsp+30h] [rbp-70h] BYREF

  LOBYTE(v16) = 0;
  if ( a6 )
  {
    *((_QWORD *)&v14 + 1) = v16;
    *(_QWORD *)&v14 = a7;
    v17 = v19;
    v18 = 0x400000000LL;
    sub_34B8C80(a1, a2, a3, a4, a5, (__int64)&v17, v14);
    v10 = v17;
    v11 = &v17[16 * (unsigned int)v18];
    if ( v11 != v17 )
    {
      v12 = *(unsigned int *)(a6 + 8);
      do
      {
        v13 = *(_QWORD *)v10;
        if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a6 + 12) )
        {
          sub_C8D5F0(a6, (const void *)(a6 + 16), v12 + 1, 8u, v8, v9);
          v12 = *(unsigned int *)(a6 + 8);
        }
        v10 += 16;
        *(_QWORD *)(*(_QWORD *)a6 + 8 * v12) = v13;
        v12 = (unsigned int)(*(_DWORD *)(a6 + 8) + 1);
        *(_DWORD *)(a6 + 8) = v12;
      }
      while ( v11 != v10 );
      v10 = v17;
    }
    if ( v10 != v19 )
      _libc_free((unsigned __int64)v10);
  }
  else
  {
    *((_QWORD *)&v15 + 1) = v16;
    *(_QWORD *)&v15 = a7;
    sub_34B8C80(a1, a2, a3, a4, a5, 0, v15);
  }
}
