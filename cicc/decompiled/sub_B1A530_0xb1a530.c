// Function: sub_B1A530
// Address: 0xb1a530
//
__int64 __fastcall sub_B1A530(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 result; // rax
  unsigned int v6; // edx
  _QWORD *v7; // rdi
  __int64 v8; // r14
  _QWORD *v9; // rsi
  __int64 v10; // r8
  __int64 v11; // r15
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rcx
  unsigned __int64 i; // rax
  __int64 v17; // [rsp+8h] [rbp-88h]
  _QWORD *v18; // [rsp+10h] [rbp-80h] BYREF
  __int64 v19; // [rsp+18h] [rbp-78h]
  _QWORD v20[14]; // [rsp+20h] [rbp-70h] BYREF

  *(_DWORD *)(a3 + 8) = 0;
  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    result = v4;
  }
  else
  {
    v4 = 0;
    result = 0;
  }
  if ( *(_DWORD *)(a1 + 32) > (unsigned int)result )
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v4);
    if ( result )
    {
      v20[0] = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v4);
      v6 = 1;
      v18 = v20;
      v7 = v20;
      v19 = 0x800000001LL;
      do
      {
        v8 = v7[v6 - 1];
        LODWORD(v19) = v6 - 1;
        v9 = *(_QWORD **)v8;
        sub_B1A4E0(a3, *(_QWORD *)v8);
        v10 = *(_QWORD *)(v8 + 24);
        v11 = *(unsigned int *)(v8 + 32);
        v12 = 8 * v11;
        v13 = (unsigned int)v19;
        v14 = v11 + (unsigned int)v19;
        if ( v14 > HIDWORD(v19) )
        {
          v9 = v20;
          v17 = v10;
          sub_C8D5F0(&v18, v20, v14, 8);
          v13 = (unsigned int)v19;
          v10 = v17;
        }
        v7 = v18;
        v15 = &v18[v13];
        if ( v12 )
        {
          for ( i = 0; i != v12; i += 8LL )
            v15[i / 8] = *(_QWORD *)(v10 + i);
          v7 = v18;
          v13 = (unsigned int)v19;
        }
        result = v11 + v13;
        LODWORD(v19) = result;
        v6 = result;
      }
      while ( (_DWORD)result );
      if ( v7 != v20 )
        return _libc_free(v7, v9);
    }
  }
  return result;
}
