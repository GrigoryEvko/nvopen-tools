// Function: sub_199E860
// Address: 0x199e860
//
__int64 __fastcall sub_199E860(__int64 a1)
{
  unsigned int v1; // eax
  unsigned int v3; // r13d
  __int64 v4; // rcx
  _QWORD *v5; // rsi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int8 v10; // cl
  __int64 v11; // rax
  _QWORD *v12; // rbx
  _QWORD *v13; // r8
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  _QWORD *v20; // [rsp+0h] [rbp-60h]
  _QWORD *v21; // [rsp+0h] [rbp-60h]
  unsigned __int8 v22; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v23; // [rsp+Fh] [rbp-51h]
  unsigned __int8 v24; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v25[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h]

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 )
  {
    v3 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = *(_QWORD *)a1;
        v25[0] = 6;
        v25[1] = 0;
        v5 = (_QWORD *)(v4 + 24LL * v1 - 24);
        v26 = v5[2];
        if ( v26 != -8 && v26 != 0 && v26 != -16 )
        {
          sub_1649AC0(v25, *v5 & 0xFFFFFFFFFFFFFFF8LL);
          v4 = *(_QWORD *)a1;
          v1 = *(_DWORD *)(a1 + 8);
        }
        v6 = v1 - 1;
        *(_DWORD *)(a1 + 8) = v6;
        v7 = (_QWORD *)(v4 + 24 * v6);
        v8 = v7[2];
        if ( v8 != -8 && v8 != 0 && v8 != -16 )
          sub_1649B30(v7);
        v9 = v26;
        if ( v26 )
        {
          if ( v26 != -8 && v26 != -16 )
            sub_1649B30(v25);
          if ( *(_BYTE *)(v9 + 16) > 0x17u )
          {
            v10 = sub_1AE9990(v9, 0);
            if ( v10 )
              break;
          }
        }
        v1 = *(_DWORD *)(a1 + 8);
        if ( !v1 )
          return v3;
      }
      v11 = 3LL * (*(_DWORD *)(v9 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(v9 + 23) & 0x40) != 0 )
      {
        v12 = *(_QWORD **)(v9 - 8);
        v13 = &v12[v11];
      }
      else
      {
        v13 = (_QWORD *)v9;
        v12 = (_QWORD *)(v9 - v11 * 8);
      }
      if ( v12 != v13 )
        break;
LABEL_24:
      v22 = v10;
      sub_15F20C0((_QWORD *)v9);
      v1 = *(_DWORD *)(a1 + 8);
      v3 = v22;
      if ( !v1 )
        return v3;
    }
    while ( 1 )
    {
      v14 = *v12;
      if ( *(_BYTE *)(*v12 + 16LL) <= 0x17u )
        goto LABEL_23;
      v15 = v12[1];
      v16 = v12[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v16 = v15;
      if ( v15 )
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
      *v12 = 0;
      if ( *(_QWORD *)(v14 + 8) )
        goto LABEL_23;
      v18 = *(_DWORD *)(a1 + 8);
      if ( v18 >= *(_DWORD *)(a1 + 12) )
      {
        v21 = v13;
        v24 = v10;
        sub_170B450(a1, 0);
        v18 = *(_DWORD *)(a1 + 8);
        v13 = v21;
        v10 = v24;
      }
      v19 = (_QWORD *)(*(_QWORD *)a1 + 24LL * v18);
      if ( v19 )
      {
        *v19 = 6;
        v19[1] = 0;
        v19[2] = v14;
        if ( v14 == -16 || v14 == -8 )
        {
          ++*(_DWORD *)(a1 + 8);
          goto LABEL_23;
        }
        v20 = v13;
        v23 = v10;
        sub_164C220((__int64)v19);
        v18 = *(_DWORD *)(a1 + 8);
        v13 = v20;
        v10 = v23;
      }
      *(_DWORD *)(a1 + 8) = v18 + 1;
LABEL_23:
      v12 += 3;
      if ( v13 == v12 )
        goto LABEL_24;
    }
  }
  return 0;
}
