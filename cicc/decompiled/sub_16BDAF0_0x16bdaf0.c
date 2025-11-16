// Function: sub_16BDAF0
// Address: 0x16bdaf0
//
void __fastcall sub_16BDAF0(__int64 *a1, unsigned int a2)
{
  int v2; // r13d
  __int64 v3; // rax
  __int64 *i; // r14
  __int64 *v5; // rsi
  __int64 v6; // r12
  int v7; // ebx
  int v8; // eax
  unsigned __int64 v9; // [rsp+10h] [rbp-E0h]
  __int64 **v10; // [rsp+20h] [rbp-D0h]
  _BYTE *v11; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v12; // [rsp+38h] [rbp-B8h]
  _BYTE v13[176]; // [rsp+40h] [rbp-B0h] BYREF

  v2 = *((_DWORD *)a1 + 4);
  v9 = a1[1];
  v3 = (__int64)_libc_calloc(a2 + 1, 8u);
  if ( !v3 )
  {
    if ( a2 == -1 )
    {
      v3 = sub_13A3880(1u);
    }
    else
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
  }
  a1[1] = v3;
  *(_QWORD *)(v3 + 8LL * a2) = -1;
  v11 = v13;
  *((_DWORD *)a1 + 4) = a2;
  *((_DWORD *)a1 + 5) = 0;
  v12 = 0x2000000000LL;
  if ( v2 )
  {
    v10 = (__int64 **)v9;
    do
    {
      for ( i = *v10; i; LODWORD(v12) = 0 )
      {
        if ( ((unsigned __int8)i & 1) != 0 )
          break;
        v5 = i;
        i = (__int64 *)*i;
        *v5 = 0;
        v6 = a1[1];
        v7 = *((_DWORD *)a1 + 4) - 1;
        v8 = (*(__int64 (__fastcall **)(__int64 *, __int64 *, _BYTE **))(*a1 + 24))(a1, v5, &v11);
        sub_16BDA20(a1, v5, (__int64 *)(v6 + 8LL * (v8 & (unsigned int)v7)));
      }
      ++v10;
    }
    while ( v10 != (__int64 **)(v9 + 8LL * (unsigned int)(v2 - 1) + 8) );
  }
  _libc_free(v9);
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
}
