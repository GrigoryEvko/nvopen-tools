// Function: sub_2E11C60
// Address: 0x2e11c60
//
__int64 __fastcall sub_2E11C60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r8
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned __int64 v7; // r13
  __int64 *v8; // rdx
  __int64 v9; // r11
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v18; // [rsp+8h] [rbp-58h]
  unsigned __int8 v20; // [rsp+1Fh] [rbp-41h]
  __int64 *v21; // [rsp+28h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 64);
  v21 = &v3[*(unsigned int *)(a2 + 72)];
  if ( v3 != v21 )
  {
    v20 = 0;
    v4 = *(__int64 **)(a2 + 64);
    while ( 1 )
    {
      v5 = *v4;
      v6 = *(_QWORD *)(*v4 + 8);
      v7 = v6 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v6 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_10;
      v8 = (__int64 *)sub_2E09D00((__int64 *)a2, *(_QWORD *)(*v4 + 8));
      v9 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
      if ( v8 != (__int64 *)v9
        && (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v8 >> 1) & 3) <= (*(_DWORD *)(v7 + 24)
                                                                                             | (unsigned int)(v6 >> 1)
                                                                                             & 3) )
      {
        v9 = (__int64)v8;
      }
      v10 = *(_QWORD *)(a1 + 8);
      v11 = *(_QWORD *)(*(_QWORD *)(v10 + 56) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF));
      if ( !v11 )
        goto LABEL_19;
      if ( (v11 & 4) != 0 )
        goto LABEL_19;
      v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v12
        || !*(_BYTE *)(v10 + 48)
        || !*(_BYTE *)(v12 + 43)
        || *(_QWORD *)a2 != v9
        && (*(_DWORD *)((*(_QWORD *)(v9 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)(v9 - 16) >> 1) & 3) >= (*(_DWORD *)(v7 + 24) | (unsigned int)(v6 >> 1) & 3) )
      {
        goto LABEL_19;
      }
      if ( (*(_BYTE *)(v5 + 8) & 6) != 0 )
        break;
      if ( (v7 | 6) == *(_QWORD *)(v9 + 8) )
        goto LABEL_8;
LABEL_10:
      if ( v21 == ++v4 )
        return v20;
    }
    v18 = v9;
    sub_2E8D840(*(_QWORD *)(v7 + 16), *(unsigned int *)(a2 + 112), 1);
    v9 = v18;
LABEL_19:
    if ( *(_QWORD *)(v9 + 8) != (v7 | 6) )
      goto LABEL_10;
    if ( (*(_BYTE *)(v5 + 8) & 6) != 0 )
    {
      v13 = *(_QWORD *)(v7 + 16);
      sub_2E8F690(v13, *(unsigned int *)(a2 + 112), *(_QWORD *)(a1 + 16), 0);
      if ( a3 && (unsigned __int8)sub_2E8B940(v13) )
      {
        v16 = *(unsigned int *)(a3 + 8);
        if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
        {
          sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + 1, 8u, v14, v15);
          v16 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v13;
        ++*(_DWORD *)(a3 + 8);
      }
    }
    else
    {
LABEL_8:
      *(_QWORD *)(v5 + 8) = 0;
      sub_2E0A580(a2, (char *)v9, 0);
    }
    v20 = 1;
    goto LABEL_10;
  }
  return 0;
}
