// Function: sub_2E15330
// Address: 0x2e15330
//
void __fastcall sub_2E15330(__int64 a1)
{
  __int64 v1; // rcx
  int v2; // r12d
  int v3; // ebx
  __int64 v4; // rdx
  __int64 *v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rax
  unsigned int v9; // eax
  unsigned int v10; // edx
  __int64 v11; // r8
  __int64 v12; // rcx
  __int64 v13; // r9
  unsigned __int64 v14; // r13
  __int64 *v15; // rax
  __int64 *v16; // rsi
  __int64 v17; // [rsp+8h] [rbp-88h]
  unsigned __int64 v18[2]; // [rsp+10h] [rbp-80h] BYREF
  _BYTE v19[112]; // [rsp+20h] [rbp-70h] BYREF

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_DWORD *)(v1 + 64);
  if ( v2 )
  {
    v3 = 0;
    while ( 1 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v1 + 56) + 16LL * (v3 & 0x7FFFFFFF) + 8);
      if ( !v8 )
        goto LABEL_5;
      if ( (*(_BYTE *)(v8 + 4) & 8) == 0 )
        break;
      while ( 1 )
      {
        v8 = *(_QWORD *)(v8 + 32);
        if ( !v8 )
          break;
        if ( (*(_BYTE *)(v8 + 4) & 8) == 0 )
          goto LABEL_9;
      }
      if ( v2 == ++v3 )
        return;
LABEL_6:
      v1 = *(_QWORD *)(a1 + 8);
    }
LABEL_9:
    v9 = *(_DWORD *)(a1 + 160);
    v10 = (v3 & 0x7FFFFFFF) + 1;
    v11 = v3 | 0x80000000;
    if ( v10 > v9 )
    {
      v12 = v9;
      if ( v10 != (unsigned __int64)v9 )
      {
        if ( v10 >= (unsigned __int64)v9 )
        {
          v13 = *(_QWORD *)(a1 + 168);
          v14 = v10 - (unsigned __int64)v9;
          if ( v10 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
          {
            v17 = *(_QWORD *)(a1 + 168);
            sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v10, 8u, v11, v13);
            v12 = *(unsigned int *)(a1 + 160);
            LODWORD(v11) = v3 | 0x80000000;
            v13 = v17;
          }
          v4 = *(_QWORD *)(a1 + 152);
          v15 = (__int64 *)(v4 + 8 * v12);
          v16 = &v15[v14];
          if ( v15 != v16 )
          {
            do
              *v15++ = v13;
            while ( v16 != v15 );
            LODWORD(v12) = *(_DWORD *)(a1 + 160);
            v4 = *(_QWORD *)(a1 + 152);
          }
          *(_DWORD *)(a1 + 160) = v14 + v12;
          goto LABEL_4;
        }
        *(_DWORD *)(a1 + 160) = v10;
      }
    }
    v4 = *(_QWORD *)(a1 + 152);
LABEL_4:
    v5 = (__int64 *)(v4 + 8LL * (v3 & 0x7FFFFFFF));
    v6 = sub_2E10F30(v11);
    *v5 = v6;
    v7 = v6;
    if ( (unsigned __int8)sub_2E11E80((_QWORD *)a1, v6) )
    {
      v18[0] = (unsigned __int64)v19;
      v18[1] = 0x800000000LL;
      sub_2E15100(a1, v7, (__int64)v18);
      if ( (_BYTE *)v18[0] != v19 )
        _libc_free(v18[0]);
    }
LABEL_5:
    if ( v2 == ++v3 )
      return;
    goto LABEL_6;
  }
}
