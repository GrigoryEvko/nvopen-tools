// Function: sub_2F623C0
// Address: 0x2f623c0
//
void __fastcall sub_2F623C0(__int64 *a1, __int64 *a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  unsigned int v6; // edx
  __int64 v7; // rbx
  __int64 v8; // r15
  unsigned int v9; // eax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  _BYTE *v15; // rdx
  __int64 v16; // rax
  _BYTE *v17; // r13
  _BYTE *v18; // r14
  char v19; // al
  char v20; // [rsp+16h] [rbp-5Ah]
  __int64 v24; // [rsp+30h] [rbp-40h]
  _BYTE *v25; // [rsp+38h] [rbp-38h]

  v4 = *a1;
  v6 = *(_DWORD *)(*a1 + 72);
  if ( v6 )
  {
    v7 = 0;
    v24 = v6;
    while ( 1 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v4 + 64) + 8 * v7) + 8LL);
      v9 = *(_DWORD *)(a1[16] + (v7 << 6));
      if ( v9 != 3 )
        break;
      sub_2E18870(a1[7], *a2, v8, a3);
      v12 = (v8 >> 1) & 3;
      v13 = a2[16] + ((unsigned __int64)**(unsigned int **)(a1[16] + (v7 << 6) + 48) << 6);
      if ( !*(_BYTE *)(v13 + 56) || *(_DWORD *)v13 )
      {
        if ( !v12 )
          goto LABEL_7;
        if ( !a4 )
        {
LABEL_21:
          v16 = *(unsigned int *)(a3 + 8);
          if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
          {
            sub_C8D5F0(a3, (const void *)(a3 + 16), v16 + 1, 8u, v10, v11);
            v16 = *(unsigned int *)(a3 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v16) = v8;
          ++*(_DWORD *)(a3 + 8);
          goto LABEL_7;
        }
        v20 = 0;
      }
      else
      {
        v20 = a4 & (v12 != 0);
        if ( !v20 )
          goto LABEL_7;
      }
      v14 = *(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v15 = *(_BYTE **)(v14 + 32);
      v25 = &v15[40 * (*(_DWORD *)(v14 + 40) & 0xFFFFFF)];
      if ( v15 != v25 )
      {
        while ( 1 )
        {
          v17 = v15;
          if ( sub_2DADC00(v15) )
            break;
          v15 = v17 + 40;
          if ( v25 == v17 + 40 )
            goto LABEL_14;
        }
        while ( v25 != v17 )
        {
          if ( *((_DWORD *)a1 + 2) == *((_DWORD *)v17 + 2) )
          {
            if ( (*(_DWORD *)v17 & 0xFFF00) != 0 )
            {
              v19 = v17[4];
              if ( (v19 & 1) != 0 && !v20 )
                v17[4] = v19 & 0xFE;
            }
            v17[3] &= ~0x40u;
          }
          if ( v17 + 40 == v25 )
            break;
          v18 = v17 + 40;
          while ( 1 )
          {
            v17 = v18;
            if ( sub_2DADC00(v18) )
              break;
            v18 += 40;
            if ( v25 == v18 )
            {
              if ( v20 )
                goto LABEL_15;
              goto LABEL_21;
            }
          }
        }
      }
LABEL_14:
      if ( !v20 )
        goto LABEL_21;
LABEL_15:
      if ( v24 == ++v7 )
        return;
LABEL_8:
      v4 = *a1;
    }
    if ( v9 > 3 )
    {
      if ( v9 - 4 <= 1 )
        BUG();
    }
    else if ( v9 - 1 <= 1 && (unsigned __int8)sub_2F60BD0((__int64)a1, v7, (__int64)a2) )
    {
      sub_2E18870(a1[7], *a1, v8, a3);
    }
LABEL_7:
    if ( v24 == ++v7 )
      return;
    goto LABEL_8;
  }
}
