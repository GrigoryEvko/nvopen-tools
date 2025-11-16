// Function: sub_12A0360
// Address: 0x12a0360
//
int __fastcall sub_12A0360(__int64 a1, unsigned int a2, __int16 a3)
{
  __int64 v5; // rax
  __int64 v6; // r13
  const char *v7; // r15
  size_t v8; // r14
  unsigned __int8 *v9; // r13
  unsigned __int8 *v10; // rdx
  const char *v11; // rdi
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned __int8 *v17; // rsi
  __int64 *v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v26[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( a2 && a3 && *(_DWORD *)(a1 + 480) != a2 )
  {
    *(_DWORD *)(a1 + 480) = a2;
    *(_WORD *)(a1 + 484) = a3;
  }
  v5 = *(_QWORD *)(a1 + 512);
  if ( *(_QWORD *)(a1 + 544) == v5 )
    return v5;
  v5 = sub_129E300(a2, 0);
  v6 = *(_QWORD *)(a1 + 544);
  if ( v6 == *(_QWORD *)(a1 + 552) )
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 568) - 8LL) + 512LL;
  if ( !v5 )
    return v5;
  v7 = *(const char **)v5;
  v8 = 0;
  if ( *(_QWORD *)v5 )
    v8 = strlen(*(const char **)v5);
  v9 = *(unsigned __int8 **)(v6 - 8);
  if ( *v9 == 15 )
  {
    v10 = v9;
    v5 = (__int64)v9;
  }
  else
  {
    v5 = *(_QWORD *)&v9[-8 * *((unsigned int *)v9 + 2)];
    if ( !v5 )
    {
      v12 = 0;
      v11 = byte_3F871B3;
      goto LABEL_16;
    }
    v10 = *(unsigned __int8 **)&v9[-8 * *((unsigned int *)v9 + 2)];
  }
  v5 = -(__int64)*(unsigned int *)(v5 + 8);
  v11 = *(const char **)&v10[8 * v5];
  if ( v11 )
  {
    v5 = sub_161E970(v11);
    v11 = (const char *)v5;
  }
  else
  {
    v12 = 0;
  }
LABEL_16:
  if ( v8 != v12 || v8 && (LODWORD(v5) = memcmp(v11, v7, v8), (_DWORD)v5) )
  {
    v13 = *v9;
    if ( (_BYTE)v13 == 19 )
    {
      v14 = *(_QWORD *)(a1 + 544);
      if ( v14 == *(_QWORD *)(a1 + 552) )
      {
        j_j___libc_free_0(v14, 512);
        v18 = (__int64 *)(*(_QWORD *)(a1 + 568) - 8LL);
        *(_QWORD *)(a1 + 568) = v18;
        v19 = *v18;
        v20 = *v18 + 512;
        *(_QWORD *)(a1 + 552) = v19;
        *(_QWORD *)(a1 + 560) = v20;
        *(_QWORD *)(a1 + 544) = v19 + 504;
        if ( *(_QWORD *)(v19 + 504) )
          sub_161E7C0(v19 + 504);
      }
      else
      {
        *(_QWORD *)(a1 + 544) = v14 - 8;
        if ( *(_QWORD *)(v14 - 8) )
          sub_161E7C0(v14 - 8);
      }
      v15 = a1 + 16;
      v16 = sub_129F850(a1, a2);
      v17 = *(unsigned __int8 **)&v9[8 * (1LL - *((unsigned int *)v9 + 2))];
    }
    else
    {
      LODWORD(v5) = v13 - 17;
      if ( (unsigned __int8)v5 > 1u )
        return v5;
      v21 = *(_QWORD *)(a1 + 544);
      if ( v21 == *(_QWORD *)(a1 + 552) )
      {
        j_j___libc_free_0(v21, 512);
        v22 = (__int64 *)(*(_QWORD *)(a1 + 568) - 8LL);
        *(_QWORD *)(a1 + 568) = v22;
        v23 = *v22;
        v24 = *v22 + 512;
        *(_QWORD *)(a1 + 552) = v23;
        *(_QWORD *)(a1 + 560) = v24;
        *(_QWORD *)(a1 + 544) = v23 + 504;
        if ( *(_QWORD *)(v23 + 504) )
          sub_161E7C0(v23 + 504);
      }
      else
      {
        *(_QWORD *)(a1 + 544) = v21 - 8;
        if ( *(_QWORD *)(v21 - 8) )
          sub_161E7C0(v21 - 8);
      }
      v15 = a1 + 16;
      v17 = v9;
      v16 = sub_129F850(a1, a2);
    }
    v26[0] = sub_15A69C0(v15, v17, v16, 0);
    LODWORD(v5) = sub_129F4F0((__int64 *)(a1 + 496), v26);
  }
  return v5;
}
