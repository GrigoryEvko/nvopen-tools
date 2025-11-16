// Function: sub_22F5820
// Address: 0x22f5820
//
void __fastcall sub_22F5820(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r8
  const void *v4; // rsi
  char *v5; // r12
  char *v6; // r14
  __int64 v7; // rcx
  _BYTE *v8; // rax
  char v9; // bl
  _BYTE *v10; // rdi
  __int64 v11; // r9
  _BYTE *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // [rsp+0h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 80);
  v2 = v1 + 16LL * *(unsigned int *)(a1 + 88);
  if ( v2 != v1 )
  {
    v4 = (const void *)(a1 + 168);
    while ( 1 )
    {
      v5 = *(char **)v1;
      v6 = (char *)(*(_QWORD *)v1 + *(_QWORD *)(v1 + 8));
      if ( *(char **)v1 != v6 )
        break;
LABEL_13:
      v1 += 16;
      if ( v2 == v1 )
        return;
    }
    while ( 1 )
    {
      v7 = *(_QWORD *)(a1 + 152);
      v8 = *(_BYTE **)(a1 + 144);
      v9 = *v5;
      v10 = &v8[v7];
      v11 = v7;
      if ( v7 >> 2 > 0 )
      {
        v12 = &v8[4 * (v7 >> 2)];
        while ( v9 != *v8 )
        {
          if ( v9 == v8[1] )
          {
            ++v8;
            goto LABEL_11;
          }
          if ( v9 == v8[2] )
          {
            v8 += 2;
            goto LABEL_11;
          }
          if ( v9 == v8[3] )
          {
            v8 += 3;
            goto LABEL_11;
          }
          v8 += 4;
          if ( v12 == v8 )
          {
            v11 = v10 - v8;
            goto LABEL_16;
          }
        }
        goto LABEL_11;
      }
LABEL_16:
      if ( v11 == 2 )
        goto LABEL_23;
      if ( v11 == 3 )
        break;
      if ( v11 != 1 )
      {
LABEL_19:
        v13 = v7 + 1;
        if ( (unsigned __int64)(v7 + 1) > *(_QWORD *)(a1 + 160) )
          goto LABEL_27;
        goto LABEL_20;
      }
LABEL_25:
      if ( v9 != *v8 )
      {
        v13 = v7 + 1;
        if ( (unsigned __int64)(v7 + 1) > *(_QWORD *)(a1 + 160) )
        {
LABEL_27:
          v14 = v2;
          sub_C8D290(a1 + 144, v4, v13, 1u, v2, v11);
          v2 = v14;
          v10 = (_BYTE *)(*(_QWORD *)(a1 + 144) + *(_QWORD *)(a1 + 152));
        }
LABEL_20:
        *v10 = v9;
        ++*(_QWORD *)(a1 + 152);
        goto LABEL_12;
      }
LABEL_11:
      if ( v10 == v8 )
        goto LABEL_19;
LABEL_12:
      if ( v6 == ++v5 )
        goto LABEL_13;
    }
    if ( v9 == *v8 )
      goto LABEL_11;
    ++v8;
LABEL_23:
    if ( v9 == *v8 )
      goto LABEL_11;
    ++v8;
    goto LABEL_25;
  }
}
