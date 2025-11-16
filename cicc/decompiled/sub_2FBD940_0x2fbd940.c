// Function: sub_2FBD940
// Address: 0x2fbd940
//
void __fastcall sub_2FBD940(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 *v5; // rdx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // rax
  signed __int64 v12; // rdx
  __int64 v13; // rdx
  int *v14; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v5 = (__int64 *)sub_2E09D00((__int64 *)v4, a2);
  if ( v5 != (__int64 *)(*(_QWORD *)v4 + 24LL * *(unsigned int *)(v4 + 8))
    && (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v5 >> 1) & 3) <= (*(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                         | (unsigned int)(a2 >> 1) & 3) )
  {
    v14 = (int *)v5[2];
    if ( v14 )
      sub_2FB7E60(a1, 0, v14);
  }
  v8 = *(_QWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( !v8 )
    goto LABEL_20;
  v15 = *(_QWORD *)(v8 + 32);
  v9 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL);
  v10 = v15 + 40LL * (unsigned int)sub_2E88FE0(*(_QWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 16));
  v11 = *(_QWORD *)(v8 + 32);
  v12 = 0xCCCCCCCCCCCCCCCDLL * ((v10 - v11) >> 3);
  if ( v12 >> 2 > 0 )
  {
    v13 = v11 + 160 * (v12 >> 2);
    while ( *(_BYTE *)v11 || (*(_WORD *)(v11 + 2) & 0xFF0) == 0 || v9 != *(_DWORD *)(v11 + 8) )
    {
      if ( !*(_BYTE *)(v11 + 40) && (*(_WORD *)(v11 + 42) & 0xFF0) != 0 && v9 == *(_DWORD *)(v11 + 48) )
      {
        if ( v10 == v11 + 40 )
          goto LABEL_20;
        return;
      }
      if ( !*(_BYTE *)(v11 + 80) && (*(_WORD *)(v11 + 82) & 0xFF0) != 0 && v9 == *(_DWORD *)(v11 + 88) )
      {
        v11 += 80;
        goto LABEL_19;
      }
      if ( !*(_BYTE *)(v11 + 120) && (*(_WORD *)(v11 + 122) & 0xFF0) != 0 && v9 == *(_DWORD *)(v11 + 128) )
      {
        v11 += 120;
        goto LABEL_19;
      }
      v11 += 160;
      if ( v11 == v13 )
      {
        v12 = 0xCCCCCCCCCCCCCCCDLL * ((v10 - v11) >> 3);
        goto LABEL_29;
      }
    }
    goto LABEL_19;
  }
LABEL_29:
  if ( v12 == 2 )
    goto LABEL_40;
  if ( v12 == 3 )
  {
    if ( !*(_BYTE *)v11 && (*(_WORD *)(v11 + 2) & 0xFF0) != 0 && v9 == *(_DWORD *)(v11 + 8) )
      goto LABEL_19;
    v11 += 40;
LABEL_40:
    if ( !*(_BYTE *)v11 && (*(_WORD *)(v11 + 2) & 0xFF0) != 0 && v9 == *(_DWORD *)(v11 + 8) )
      goto LABEL_19;
    v11 += 40;
    goto LABEL_32;
  }
  if ( v12 != 1 )
    goto LABEL_20;
LABEL_32:
  if ( *(_BYTE *)v11 || (*(_WORD *)(v11 + 2) & 0xFF0) == 0 || v9 != *(_DWORD *)(v11 + 8) )
    goto LABEL_20;
LABEL_19:
  if ( v10 == v11 )
LABEL_20:
    sub_2FBD6E0(a1 + 192, a2, a3, *(unsigned int *)(a1 + 80), v6, v7);
}
