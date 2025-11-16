// Function: sub_3852EE0
// Address: 0x3852ee0
//
__int64 __fastcall sub_3852EE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rbx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r12
  int v15; // edx
  int v16; // edx
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r11
  unsigned int v21; // r12d
  __int64 v23; // rbx
  int v24; // eax
  int v25; // edi
  __int64 *v26; // rsi
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v29; // [rsp+18h] [rbp-68h]
  __int64 v30; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v31; // [rsp+30h] [rbp-50h] BYREF
  __int64 v32; // [rsp+38h] [rbp-48h]
  _BYTE v33[64]; // [rsp+40h] [rbp-40h] BYREF

  v32 = 0x200000000LL;
  v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v31 = (__int64 *)v33;
  v10 = 24 * v9;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v11 = *(_QWORD *)(a2 - 8);
    v12 = v11 + v10;
  }
  else
  {
    v11 = a2 - v10;
    v12 = a2;
  }
  if ( v11 == v12 )
  {
    v26 = (__int64 *)v33;
  }
  else
  {
    do
    {
      v14 = *(_QWORD *)v11;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 16LL) > 0x10u )
      {
        v15 = *(_DWORD *)(a1 + 160);
        if ( !v15 )
          goto LABEL_12;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 144);
        v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v19 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v19;
        if ( *v19 != v14 )
        {
          v24 = 1;
          while ( v20 != -8 )
          {
            v25 = v24 + 1;
            v18 = v16 & (v24 + v18);
            v19 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v19;
            if ( v14 == *v19 )
              goto LABEL_11;
            v24 = v25;
          }
LABEL_12:
          v21 = 0;
          goto LABEL_13;
        }
LABEL_11:
        v14 = v19[1];
        if ( !v14 )
          goto LABEL_12;
      }
      v13 = (unsigned int)v32;
      if ( (unsigned int)v32 >= HIDWORD(v32) )
      {
        v28 = a3;
        v29 = v12;
        sub_16CD150((__int64)&v31, v33, 0, 8, v12, a6);
        v13 = (unsigned int)v32;
        a3 = v28;
        v12 = v29;
      }
      v11 += 24;
      v31[v13] = v14;
      LODWORD(v32) = v32 + 1;
    }
    while ( v12 != v11 );
    v26 = v31;
  }
  v21 = 0;
  v23 = sub_14DD1F0(a3, v26, 1, *(_BYTE **)(a4 + 40), 0);
  if ( v23 )
  {
    v30 = a2;
    v21 = 1;
    sub_38526A0(a1 + 136, &v30)[1] = v23;
  }
LABEL_13:
  if ( v31 != (__int64 *)v33 )
    _libc_free((unsigned __int64)v31);
  return v21;
}
