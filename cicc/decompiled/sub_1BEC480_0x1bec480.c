// Function: sub_1BEC480
// Address: 0x1bec480
//
void __fastcall sub_1BEC480(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  _QWORD *v4; // rax
  int v5; // r8d
  int v6; // r9d
  __int64 v7; // r12
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 *v10; // rdi
  __int64 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 *v14; // r12
  __int64 v15; // r13
  __int64 v16; // [rsp+8h] [rbp-88h]
  __int64 *v17; // [rsp+10h] [rbp-80h] BYREF
  __int64 v18; // [rsp+18h] [rbp-78h]
  _BYTE v19[112]; // [rsp+20h] [rbp-70h] BYREF

  v3 = *(_QWORD *)(a3 + 8);
  v17 = (__int64 *)v19;
  v18 = 0x800000000LL;
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = sub_1648700(v3);
      if ( (unsigned __int8)(*((_BYTE *)v4 + 16) - 25) <= 9u )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return;
    }
    while ( 1 )
    {
      v7 = sub_1BEC0C0(a1, v4[5]);
      v8 = (unsigned int)v18;
      if ( (unsigned int)v18 >= HIDWORD(v18) )
      {
        sub_16CD150((__int64)&v17, v19, 0, 8, v5, v6);
        v8 = (unsigned int)v18;
      }
      v17[v8] = v7;
      v9 = (unsigned int)(v18 + 1);
      LODWORD(v18) = v18 + 1;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        break;
      while ( 1 )
      {
        v4 = sub_1648700(v3);
        if ( (unsigned __int8)(*((_BYTE *)v4 + 16) - 25) <= 9u )
          break;
        v3 = *(_QWORD *)(v3 + 8);
        if ( !v3 )
          goto LABEL_8;
      }
    }
LABEL_8:
    v10 = v17;
    v11 = &v17[v9];
    if ( v11 != v17 )
    {
      v12 = *(unsigned int *)(a2 + 64);
      v13 = a2 + 56;
      v14 = v17;
      do
      {
        v15 = *v14;
        if ( *(_DWORD *)(a2 + 68) <= (unsigned int)v12 )
        {
          v16 = v13;
          sub_16CD150(v13, (const void *)(a2 + 72), 0, 8, v13, v6);
          v12 = *(unsigned int *)(a2 + 64);
          v13 = v16;
        }
        ++v14;
        *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * v12) = v15;
        v12 = (unsigned int)(*(_DWORD *)(a2 + 64) + 1);
        *(_DWORD *)(a2 + 64) = v12;
      }
      while ( v11 != v14 );
      v10 = v17;
    }
    if ( v10 != (__int64 *)v19 )
      _libc_free((unsigned __int64)v10);
  }
}
