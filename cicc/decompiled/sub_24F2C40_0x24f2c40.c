// Function: sub_24F2C40
// Address: 0x24f2c40
//
__int64 __fastcall sub_24F2C40(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 *i; // r14
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // r11
  _QWORD *v11; // rax
  unsigned int v12; // ecx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // [rsp+10h] [rbp-A0h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int64 v20; // [rsp+18h] [rbp-98h]
  __int64 v21; // [rsp+20h] [rbp-90h]
  __int64 v22; // [rsp+2Ch] [rbp-84h]
  __int64 v23; // [rsp+48h] [rbp-68h] BYREF
  _BYTE v24[32]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v25; // [rsp+70h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 16);
  v2 = *(__int64 **)(v1 + 120);
  result = *(unsigned int *)(v1 + 128);
  for ( i = &v2[result]; i != v2; ++v2 )
  {
    v5 = *v2;
    v6 = *(_QWORD *)(*v2 - 32);
    if ( !v6 || *(_BYTE *)v6 || *(_QWORD *)(v6 + 24) != *(_QWORD *)(v5 + 80) )
      BUG();
    if ( *(_DWORD *)(v6 + 36) != 60 )
      sub_C64ED0("coro.id must be paired with coro.suspend", 1u);
    v7 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
    if ( *(_BYTE *)v7 == 85 )
    {
      v16 = *(_QWORD *)(v7 - 32);
      if ( v16 )
      {
        if ( !*(_BYTE *)v16 )
        {
          result = *(_QWORD *)(v7 + 80);
          if ( *(_QWORD *)(v16 + 24) == result && (*(_BYTE *)(v16 + 33) & 0x20) != 0 && *(_DWORD *)(v16 + 36) == 57 )
            continue;
        }
      }
    }
    v19 = **(_QWORD **)(a1 + 16);
    v8 = (__int64 *)sub_B43CA0(*v2);
    v9 = sub_B6E160(v8, 0x39u, 0, 0);
    v10 = 0;
    v25 = 257;
    v23 = v19;
    if ( v9 )
      v10 = *(_QWORD *)(v9 + 24);
    v20 = v10;
    v18 = v9;
    v11 = sub_BD2C40(88, 2u);
    if ( v11 )
    {
      v21 = (__int64)v11;
      v12 = v22 & 0xE0000000 | 2;
      v22 = v12;
      sub_B44260((__int64)v11, **(_QWORD **)(v20 + 16), 56, v12, v5 + 24, 0);
      *(_QWORD *)(v21 + 72) = 0;
      sub_B4A290(v21, v20, v18, &v23, 1, (__int64)v24, HIDWORD(v22), 0);
      result = v21;
      v13 = v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
      if ( *(_QWORD *)v13 )
      {
        v14 = *(_QWORD *)(v13 + 8);
        **(_QWORD **)(v13 + 16) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = *(_QWORD *)(v13 + 16);
      }
      *(_QWORD *)v13 = v21;
      v15 = *(_QWORD *)(v21 + 16);
      *(_QWORD *)(v13 + 8) = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = v13 + 8;
      *(_QWORD *)(v13 + 16) = v21 + 16;
      *(_QWORD *)(v21 + 16) = v13;
    }
    else
    {
      result = -32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF);
      v17 = result + v5;
      if ( *(_QWORD *)v17 )
      {
        result = *(_QWORD *)(v17 + 8);
        **(_QWORD **)(v17 + 16) = result;
        if ( result )
          *(_QWORD *)(result + 16) = *(_QWORD *)(v17 + 16);
        *(_QWORD *)v17 = 0;
      }
    }
  }
  return result;
}
