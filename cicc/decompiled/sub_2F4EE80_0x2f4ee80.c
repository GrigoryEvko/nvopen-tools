// Function: sub_2F4EE80
// Address: 0x2f4ee80
//
__int64 __fastcall sub_2F4EE80(__int64 a1, int a2)
{
  unsigned int v2; // eax
  __int64 v3; // r8
  __int64 v4; // r15
  __int64 v6; // r14
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r12
  unsigned int v11; // eax
  __int64 v12; // rcx
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // r9
  _QWORD *v17; // rax
  _QWORD *v18; // rsi
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v2 = a2 & 0x7FFFFFFF;
  v3 = a2 & 0x7FFFFFFF;
  v4 = 8 * v3;
  v6 = *(_QWORD *)(a1 - 728);
  v7 = *(unsigned int *)(v6 + 160);
  if ( (a2 & 0x7FFFFFFFu) < (unsigned int)v7 )
  {
    v8 = *(_QWORD *)(v6 + 152);
    v9 = *(_QWORD *)(v8 + 8LL * v2);
    if ( v9 )
      goto LABEL_3;
  }
  v11 = v2 + 1;
  if ( (unsigned int)v7 < v11 && v11 != v7 )
  {
    if ( v11 >= v7 )
    {
      v15 = *(_QWORD *)(v6 + 168);
      v16 = v11 - v7;
      if ( v11 > (unsigned __int64)*(unsigned int *)(v6 + 164) )
      {
        v20 = v11 - v7;
        sub_C8D5F0(v6 + 152, (const void *)(v6 + 168), v11, 8u, v3, v16);
        v7 = *(unsigned int *)(v6 + 160);
        v3 = a2 & 0x7FFFFFFF;
        v16 = v20;
      }
      v12 = *(_QWORD *)(v6 + 152);
      v17 = (_QWORD *)(v12 + 8 * v7);
      v18 = &v17[v16];
      if ( v17 != v18 )
      {
        do
          *v17++ = v15;
        while ( v18 != v17 );
        LODWORD(v7) = *(_DWORD *)(v6 + 160);
        v12 = *(_QWORD *)(v6 + 152);
      }
      *(_DWORD *)(v6 + 160) = v16 + v7;
      goto LABEL_7;
    }
    *(_DWORD *)(v6 + 160) = v11;
  }
  v12 = *(_QWORD *)(v6 + 152);
LABEL_7:
  v19 = v3;
  v13 = (__int64 *)(v12 + v4);
  v14 = sub_2E10F30(a2);
  *v13 = v14;
  v9 = v14;
  sub_2E11E80((_QWORD *)v6, v14);
  v3 = v19;
LABEL_3:
  if ( *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 - 736) + 32LL) + 4 * v3) )
  {
    sub_2E21040(*(_QWORD **)(a1 - 720), v9, v7, v8, v3);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)(a1 - 760) + 64LL))(a1 - 760, v9);
    return 1;
  }
  else
  {
    *(_DWORD *)(v9 + 72) = 0;
    *(_DWORD *)(v9 + 8) = 0;
    return 0;
  }
}
