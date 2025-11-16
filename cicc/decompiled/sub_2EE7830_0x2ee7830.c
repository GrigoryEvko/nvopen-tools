// Function: sub_2EE7830
// Address: 0x2ee7830
//
__int64 __fastcall sub_2EE7830(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r14
  unsigned int v5; // r13d
  __int64 v6; // r12
  char v7; // al
  unsigned int v8; // edx
  __int64 v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // r9
  _QWORD *v16; // rax
  unsigned int v18; // [rsp+0h] [rbp-40h]
  unsigned __int64 v19; // [rsp+0h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 32);
  v4 = v3 + 40LL * (*(_DWORD *)(a1 + 40) & 0xFFFFFF);
  if ( v3 != v4 )
  {
    v5 = 0;
    while ( 1 )
    {
      if ( *(_BYTE *)v3 )
        goto LABEL_18;
      v6 = *(unsigned int *)(v3 + 8);
      if ( !(_DWORD)v6 )
        goto LABEL_18;
      if ( (unsigned int)(v6 - 1) <= 0x3FFFFFFE )
      {
        v3 += 40;
        v5 = 1;
        if ( v4 == v3 )
          return v5;
      }
      else
      {
        v7 = *(_BYTE *)(v3 + 4);
        if ( (v7 & 1) == 0 && (v7 & 2) == 0 )
        {
          if ( (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
          {
            v8 = sub_2EAB0A0(v3);
            if ( (int)v6 < 0 )
              goto LABEL_22;
            goto LABEL_10;
          }
          if ( (*(_DWORD *)v3 & 0xFFF00) != 0 )
          {
            v8 = sub_2EAB0A0(v3);
            if ( (int)v6 < 0 )
            {
LABEL_22:
              v9 = *(_QWORD *)(*(_QWORD *)(a3 + 56) + 16 * (v6 & 0x7FFFFFFF) + 8);
              goto LABEL_11;
            }
LABEL_10:
            v9 = *(_QWORD *)(*(_QWORD *)(a3 + 304) + 8 * v6);
LABEL_11:
            if ( !v9
              || (*(_BYTE *)(v9 + 3) & 0x10) == 0
              && ((v9 = *(_QWORD *)(v9 + 32)) == 0 || (*(_BYTE *)(v9 + 3) & 0x10) == 0)
              || (v10 = *(_QWORD *)(v9 + 32)) != 0 && (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
            {
              BUG();
            }
            v18 = v8;
            v11 = *(_QWORD *)(v9 + 16);
            v13 = (unsigned int)sub_2EAB0A0(v9);
            v14 = *(unsigned int *)(a2 + 8);
            v15 = ((unsigned __int64)v18 << 32) | v13;
            if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
            {
              v19 = v15;
              sub_C8D5F0(a2, (const void *)(a2 + 16), v14 + 1, 0x10u, v12, v15);
              v14 = *(unsigned int *)(a2 + 8);
              v15 = v19;
            }
            v16 = (_QWORD *)(*(_QWORD *)a2 + 16 * v14);
            *v16 = v11;
            v16[1] = v15;
            ++*(_DWORD *)(a2 + 8);
          }
        }
LABEL_18:
        v3 += 40;
        if ( v4 == v3 )
          return v5;
      }
    }
  }
  return 0;
}
