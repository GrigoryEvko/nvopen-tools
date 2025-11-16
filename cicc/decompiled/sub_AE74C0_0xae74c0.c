// Function: sub_AE74C0
// Address: 0xae74c0
//
unsigned __int64 *__fastcall sub_AE74C0(unsigned __int64 *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // r8
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  unsigned __int64 v12; // r10
  __int64 v13; // rax
  unsigned __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rax
  unsigned __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int64 v18; // [rsp+18h] [rbp-38h]
  unsigned __int64 v19; // [rsp+18h] [rbp-38h]
  unsigned __int64 v20; // [rsp+18h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 7) & 8) != 0 && sub_B91390(a2) && (v4 = sub_BD5C60(a2, a2, v3), (v5 = sub_B9F8A0(v4)) != 0) )
  {
    v6 = *(_QWORD *)(v5 + 16);
    v7 = 0;
    if ( v6 )
    {
      while ( 1 )
      {
        v8 = *(_QWORD *)(v6 + 24);
        if ( *(_BYTE *)v8 != 85 )
          goto LABEL_8;
        v9 = *(_QWORD *)(v8 - 32);
        if ( !v9
          || *(_BYTE *)v9
          || *(_QWORD *)(v9 + 24) != *(_QWORD *)(v8 + 80)
          || (*(_BYTE *)(v9 + 33) & 0x20) == 0
          || *(_DWORD *)(v9 + 36) != 69 )
        {
          goto LABEL_8;
        }
        if ( (v7 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (v7 & 4) != 0 )
          {
            v10 = *(unsigned int *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 8);
            v11 = v7 & 0xFFFFFFFFFFFFFFF8LL;
            v12 = v7 & 0xFFFFFFFFFFFFFFF8LL;
          }
          else
          {
            v18 = v7 & 0xFFFFFFFFFFFFFFF8LL;
            v13 = sub_22077B0(48);
            v14 = v18;
            if ( v13 )
            {
              *(_QWORD *)(v13 + 8) = 0x400000000LL;
              *(_QWORD *)v13 = v13 + 16;
            }
            v15 = v13;
            v11 = v13 & 0xFFFFFFFFFFFFFFF8LL;
            v16 = *(unsigned int *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 8);
            v7 = v15 | 4;
            v12 = v11;
            if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
            {
              v17 = v18;
              v20 = v7;
              sub_C8D5F0(v11, v11 + 16, v16 + 1, 8);
              v16 = *(unsigned int *)(v11 + 8);
              v14 = v17;
              v12 = v11;
              v7 = v20;
            }
            *(_QWORD *)(*(_QWORD *)v11 + 8 * v16) = v14;
            v10 = (unsigned int)(*(_DWORD *)(v11 + 8) + 1);
            *(_DWORD *)(v11 + 8) = v10;
          }
          if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
          {
            v19 = v7;
            sub_C8D5F0(v12, v11 + 16, v10 + 1, 8);
            v10 = *(unsigned int *)(v11 + 8);
            v7 = v19;
          }
          *(_QWORD *)(*(_QWORD *)v11 + 8 * v10) = v8;
          ++*(_DWORD *)(v11 + 8);
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            break;
        }
        else
        {
          v7 = v8 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_8:
          v6 = *(_QWORD *)(v6 + 8);
          if ( !v6 )
            break;
        }
      }
    }
    *a1 = v7;
  }
  else
  {
    *a1 = 0;
  }
  return a1;
}
