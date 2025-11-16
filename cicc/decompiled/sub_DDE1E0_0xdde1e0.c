// Function: sub_DDE1E0
// Address: 0xdde1e0
//
__int64 __fastcall sub_DDE1E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int16 v6; // bx
  unsigned int v7; // r13d
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rax
  char *v12; // r15
  __int64 v13; // rax
  bool v14; // al
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 *v17; // r9
  __int64 v18; // r8
  _BYTE *v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-48h]
  _BYTE *v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned int v25[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = *(_WORD *)(a2 + 28) & 7;
  v7 = *(_WORD *)(a2 + 28) & 7;
  if ( (*(_WORD *)(a2 + 28) & 4) != 0 || *(_QWORD *)(a2 + 40) != 2 )
    return v7;
  if ( !*(_BYTE *)(a1 + 1412) )
    goto LABEL_11;
  v9 = *(__int64 **)(a1 + 1392);
  a4 = *(unsigned int *)(a1 + 1404);
  a3 = &v9[a4];
  if ( v9 == a3 )
  {
LABEL_10:
    if ( (unsigned int)a4 < *(_DWORD *)(a1 + 1400) )
    {
      v10 = (unsigned int)(a4 + 1);
      *(_DWORD *)(a1 + 1404) = v10;
      *a3 = a2;
      ++*(_QWORD *)(a1 + 1384);
LABEL_12:
      v11 = sub_D33D80((_QWORD *)a2, a1, (__int64)a3, v10, a5);
      v12 = *(char **)(a2 + 48);
      v22 = v11;
      v13 = sub_DCF3A0((__int64 *)a1, v12, 1);
      v14 = sub_D96A50(v13);
      v18 = v22;
      if ( !v14 || *(_BYTE *)(a1 + 16) )
        goto LABEL_14;
      v20 = *(_QWORD *)(a1 + 32);
      if ( !*(_BYTE *)(v20 + 192) )
      {
        v21 = v22;
        v24 = *(_QWORD *)(a1 + 32);
        sub_CFDFC0(v24, (__int64)v12, v15, v16, v18, v17);
        v18 = v21;
        v20 = v24;
      }
      if ( *(_DWORD *)(v20 + 24) )
      {
LABEL_14:
        v19 = (_BYTE *)sub_DBF6A0(v18, v25, (__int64 *)a1);
        if ( v19 )
        {
          v23 = v19;
          if ( (unsigned __int8)sub_DDDA00(a1, (__int64)v12, v25[0], a2, v19)
            || (unsigned __int8)sub_DDDEB0((__int64 *)a1, v25[0], a2, v23) )
          {
            return v6 | 4u;
          }
        }
      }
      return v7;
    }
LABEL_11:
    sub_C8CC70(a1 + 1384, a2, (__int64)a3, a4, a5, a6);
    if ( !(_BYTE)a3 )
      return v7;
    goto LABEL_12;
  }
  while ( a2 != *v9 )
  {
    if ( a3 == ++v9 )
      goto LABEL_10;
  }
  return v7;
}
