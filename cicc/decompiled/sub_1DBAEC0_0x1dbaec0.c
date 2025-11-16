// Function: sub_1DBAEC0
// Address: 0x1dbaec0
//
__int64 __fastcall sub_1DBAEC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r8
  __int64 *v5; // r14
  unsigned int v6; // eax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rbx
  unsigned __int64 v10; // r13
  _QWORD *v11; // rsi
  _QWORD *v12; // r11
  __int64 v13; // rdi
  __int64 v14; // rbx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // rax
  _QWORD *v19; // [rsp+8h] [rbp-58h]
  unsigned __int8 v21; // [rsp+1Fh] [rbp-41h]
  __int64 *v23; // [rsp+28h] [rbp-38h]

  v3 = *(__int64 **)(a2 + 64);
  v23 = &v3[*(unsigned int *)(a2 + 72)];
  if ( v3 != v23 )
  {
    v21 = 0;
    v5 = *(__int64 **)(a2 + 64);
    while ( 1 )
    {
      v8 = *v5;
      v9 = *(_QWORD *)(*v5 + 8);
      v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_11;
      v11 = (_QWORD *)sub_1DB3C70((__int64 *)a2, *(_QWORD *)(*v5 + 8));
      v12 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
      if ( v11 != v12
        && (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*v11 >> 1) & 3) <= (*(_DWORD *)(v10 + 24) | (unsigned int)(v9 >> 1) & 3) )
      {
        v12 = v11;
      }
      v13 = *(_QWORD *)(a1 + 240);
      if ( *(_BYTE *)(v13 + 16)
        && *(_BYTE *)((*(_QWORD *)(*(_QWORD *)(v13 + 24) + 16LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF))
                     & 0xFFFFFFFFFFFFFFF8LL)
                    + 29)
        && (*(_QWORD **)a2 == v12
         || (*(_DWORD *)((*(v12 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)*(v12 - 2) >> 1) & 3) < (*(_DWORD *)(v10 + 24) | (unsigned int)(v9 >> 1) & 3)) )
      {
        if ( (*(_BYTE *)(v8 + 8) & 6) == 0 )
        {
          if ( v12[1] == (v10 | 6) )
            goto LABEL_8;
          goto LABEL_11;
        }
        v19 = v12;
        sub_1E1A650(*(_QWORD *)(v10 + 16), *(unsigned int *)(a2 + 112), 1);
        v12 = v19;
      }
      if ( v12[1] != (v10 | 6) )
        goto LABEL_11;
      if ( (*(_BYTE *)(v8 + 8) & 6) == 0 )
      {
LABEL_8:
        *(_QWORD *)(v8 + 8) = 0;
        v6 = *(_DWORD *)(a2 + 8);
        v7 = *(_QWORD *)a2 + 24LL * v6;
        if ( (_QWORD *)v7 != v12 + 3 )
        {
          memmove(v12, v12 + 3, v7 - (_QWORD)(v12 + 3));
          v6 = *(_DWORD *)(a2 + 8);
        }
        v21 = 1;
        *(_DWORD *)(a2 + 8) = v6 - 1;
        goto LABEL_11;
      }
      v14 = *(_QWORD *)(v10 + 16);
      sub_1E1B440(v14, *(unsigned int *)(a2 + 112), *(_QWORD *)(a1 + 248), 0);
      if ( a3 && (unsigned __int8)sub_1E17E50(v14) )
      {
        v17 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v17 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v15, v16);
          v17 = *(unsigned int *)(a3 + 8);
        }
        ++v5;
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v17) = v14;
        ++*(_DWORD *)(a3 + 8);
        if ( v23 == v5 )
          return v21;
      }
      else
      {
LABEL_11:
        if ( v23 == ++v5 )
          return v21;
      }
    }
  }
  return 0;
}
