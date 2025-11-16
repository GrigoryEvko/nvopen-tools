// Function: sub_1A3F630
// Address: 0x1a3f630
//
void __fastcall sub_1A3F630(__int64 a1, __int64 a2, __int64 a3)
{
  bool v4; // sf
  _BYTE *v5; // rax
  unsigned int i; // ebx
  __int64 v7; // r15
  _BYTE *v8; // r14
  __int64 v9; // r8
  _BYTE *v10; // r12
  __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rsi
  unsigned __int8 *v16; // rsi
  int v18; // [rsp+24h] [rbp-9Ch]
  unsigned __int8 *v19; // [rsp+38h] [rbp-88h] BYREF
  _BYTE *v20; // [rsp+40h] [rbp-80h] BYREF
  __int64 v21; // [rsp+48h] [rbp-78h]
  _BYTE v22[112]; // [rsp+50h] [rbp-70h] BYREF

  v20 = v22;
  v4 = *(__int16 *)(a2 + 18) < 0;
  v21 = 0x400000000LL;
  if ( !v4 )
  {
    v18 = *(_DWORD *)(a3 + 8);
    if ( !v18 )
      return;
    v5 = v22;
    goto LABEL_4;
  }
  sub_161F980(a2, (__int64)&v20);
  v18 = *(_DWORD *)(a3 + 8);
  v5 = v20;
  if ( v18 )
  {
LABEL_4:
    for ( i = 0; i != v18; ++i )
    {
      v7 = *(_QWORD *)(*(_QWORD *)a3 + 8LL * i);
      if ( *(_BYTE *)(v7 + 16) > 0x17u )
      {
        v8 = v5;
        v9 = 16LL * (unsigned int)v21;
        v10 = &v5[v9];
        if ( &v5[v9] != v5 )
        {
          do
          {
            while ( 1 )
            {
              v12 = *(_DWORD *)v8;
              if ( (*(_DWORD *)v8 & 0xFFFFFFFD) != 1 && (unsigned int)(v12 - 5) > 3 && v12 != *(_DWORD *)(a1 + 512) )
                break;
              v11 = *((_QWORD *)v8 + 1);
              v8 += 16;
              sub_1625C10(v7, v12, v11);
              if ( v10 == v8 )
                goto LABEL_13;
            }
            v8 += 16;
          }
          while ( v10 != v8 );
        }
LABEL_13:
        v13 = *(_QWORD *)(a2 + 48);
        if ( v13 && !*(_QWORD *)(v7 + 48) )
        {
          v19 = *(unsigned __int8 **)(a2 + 48);
          sub_1623A60((__int64)&v19, v13, 2);
          v14 = v7 + 48;
          if ( (unsigned __int8 **)(v7 + 48) == &v19 )
          {
            if ( v19 )
              sub_161E7C0((__int64)&v19, (__int64)v19);
          }
          else
          {
            v15 = *(_QWORD *)(v7 + 48);
            if ( v15 )
            {
              sub_161E7C0(v7 + 48, v15);
              v14 = v7 + 48;
            }
            v16 = v19;
            *(_QWORD *)(v7 + 48) = v19;
            if ( v16 )
              sub_1623210((__int64)&v19, v16, v14);
          }
        }
        v5 = v20;
      }
    }
  }
  if ( v5 != v22 )
    _libc_free((unsigned __int64)v5);
}
