// Function: sub_3099AF0
// Address: 0x3099af0
//
void __fastcall sub_3099AF0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rbx
  __int64 v4; // r12
  const char *v5; // rax
  size_t v6; // rdx
  size_t v7; // r12
  int v8; // eax
  unsigned int v9; // r8d
  _QWORD *v10; // r9
  __int64 v11; // rax
  unsigned int v12; // r8d
  _QWORD *v13; // r9
  _QWORD *v14; // rcx
  _QWORD *v15; // [rsp+0h] [rbp-50h]
  _QWORD *v16; // [rsp+8h] [rbp-48h]
  unsigned int v17; // [rsp+14h] [rbp-3Ch]
  const char *src; // [rsp+18h] [rbp-38h]

  v2 = a1 + 24;
  v3 = *(_QWORD *)(a1 + 32);
  if ( v3 != a1 + 24 )
  {
    while ( 1 )
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      if ( sub_B2FC80(v4) || (*(_BYTE *)(v4 + 32) & 0xF) != 0 )
        goto LABEL_3;
      v5 = sub_BD5D20(v4);
      v7 = v6;
      src = v5;
      v8 = sub_C92610();
      v9 = sub_C92740(a2, src, v7, v8);
      v10 = (_QWORD *)(*(_QWORD *)a2 + 8LL * v9);
      if ( *v10 )
      {
        if ( *v10 == -8 )
        {
          --*(_DWORD *)(a2 + 16);
          goto LABEL_11;
        }
LABEL_3:
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
      else
      {
LABEL_11:
        v16 = v10;
        v17 = v9;
        v11 = sub_C7D670(v7 + 9, 8);
        v12 = v17;
        v13 = v16;
        v14 = (_QWORD *)v11;
        if ( v7 )
        {
          v15 = (_QWORD *)v11;
          memcpy((void *)(v11 + 8), src, v7);
          v12 = v17;
          v13 = v16;
          v14 = v15;
        }
        *((_BYTE *)v14 + v7 + 8) = 0;
        *v14 = v7;
        *v13 = v14;
        ++*(_DWORD *)(a2 + 12);
        sub_C929D0((__int64 *)a2, v12);
        v3 = *(_QWORD *)(v3 + 8);
        if ( v2 == v3 )
          return;
      }
    }
  }
}
