// Function: sub_185BE50
// Address: 0x185be50
//
__int64 __fastcall sub_185BE50(__int128 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  int v7; // r13d
  unsigned int i; // r15d
  int v9; // r8d
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 *v12; // rcx
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v17; // r15
  __int64 j; // r13
  int v19; // r8d
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 *v22; // rdx
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // [rsp+0h] [rbp-170h]
  __int64 v26; // [rsp+0h] [rbp-170h]
  __int64 v27; // [rsp+10h] [rbp-160h]
  __int64 v28; // [rsp+10h] [rbp-160h]
  __int64 *v30; // [rsp+30h] [rbp-140h] BYREF
  __int64 v31; // [rsp+38h] [rbp-138h]
  _BYTE v32[304]; // [rsp+40h] [rbp-130h] BYREF

  v3 = *((_QWORD *)&a1 + 1);
  v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  if ( a3 != (_DWORD)v4 )
  {
    v5 = a1;
    *(_QWORD *)&a1 = *(_QWORD *)a1;
    v30 = (__int64 *)v32;
    v31 = 0x2000000000LL;
    v27 = a3;
    if ( *(_BYTE *)(a1 + 8) == 13 )
    {
      v7 = *(_DWORD *)(a1 + 12);
      if ( v7 )
      {
        for ( i = 0; i != v7; ++i )
        {
          v10 = sub_15A0A60(v5, i);
          v11 = (unsigned int)v31;
          if ( (unsigned int)v31 >= HIDWORD(v31) )
          {
            v25 = v10;
            sub_16CD150((__int64)&v30, v32, 0, 8, v9, v10);
            v11 = (unsigned int)v31;
            v10 = v25;
          }
          v30[v11] = v10;
          LODWORD(v31) = v31 + 1;
        }
        v12 = v30;
        v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      }
      else
      {
        v12 = (__int64 *)v32;
      }
      v13 = *(_QWORD *)(a2 + 24 * (v27 - v4));
      v14 = *(_QWORD **)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
        v14 = (_QWORD *)*v14;
      v12[(unsigned int)v14] = sub_185BE50(v12[(unsigned int)v14], *((_QWORD *)&a1 + 1), a2, a3 + 1);
      v3 = sub_159F090((__int64 **)a1, v30, (unsigned int)v31, v15);
    }
    else
    {
      v28 = *(_QWORD *)(a2 + 24 * (a3 - v4));
      v17 = *(_QWORD *)(a1 + 32);
      if ( v17 )
      {
        for ( j = 0; j != v17; ++j )
        {
          v20 = sub_15A0A60(v5, j);
          v21 = (unsigned int)v31;
          if ( (unsigned int)v31 >= HIDWORD(v31) )
          {
            v26 = v20;
            sub_16CD150((__int64)&v30, v32, 0, 8, v19, v20);
            v21 = (unsigned int)v31;
            v20 = v26;
          }
          v30[v21] = v20;
          LODWORD(v31) = v31 + 1;
        }
        v22 = v30;
      }
      else
      {
        v22 = (__int64 *)v32;
      }
      v23 = *(_QWORD **)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) > 0x40u )
        v23 = (_QWORD *)*v23;
      v22[(_QWORD)v23] = sub_185BE50(v22[(_QWORD)v23], *((_QWORD *)&a1 + 1), a2, a3 + 1);
      if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 14 )
      {
        *((_QWORD *)&a1 + 1) = v30;
        v3 = sub_159DFD0(a1, (unsigned int)v31, v24);
      }
      else
      {
        v3 = sub_15A01B0(v30, (unsigned int)v31);
      }
    }
    if ( v30 != (__int64 *)v32 )
      _libc_free((unsigned __int64)v30);
  }
  return v3;
}
