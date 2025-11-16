// Function: sub_3136910
// Address: 0x3136910
//
void __fastcall sub_3136910(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  bool v8; // zf
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rbx
  unsigned __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rbx
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  _QWORD *v21; // rbx
  unsigned __int64 v22; // rax
  __int64 v23; // r12
  int v24; // ebx
  unsigned int v25; // r14d
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r13
  __int64 *v29; // rax
  char v30; // dl
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // [rsp+18h] [rbp-148h]
  _BYTE *v34; // [rsp+20h] [rbp-140h] BYREF
  __int64 v35; // [rsp+28h] [rbp-138h]
  _BYTE v36[304]; // [rsp+30h] [rbp-130h] BYREF

  v8 = *(_BYTE *)(a2 + 28) == 0;
  v34 = v36;
  v9 = *(_QWORD *)(a1 + 32);
  v33 = a3;
  v35 = 0x2000000000LL;
  if ( v8 )
    goto LABEL_8;
  a3 = *(unsigned int *)(a2 + 20);
  v10 = *(__int64 **)(a2 + 8);
  v11 = &v10[a3];
  a4 = a3;
  if ( v10 != v11 )
  {
    a3 = *(_QWORD *)(a2 + 8);
    while ( v9 != *(_QWORD *)a3 )
    {
      a3 += 8;
      if ( v11 == (__int64 *)a3 )
        goto LABEL_7;
    }
    v12 = *(_QWORD *)(a1 + 40);
    goto LABEL_11;
  }
LABEL_7:
  if ( (unsigned int)a4 >= *(_DWORD *)(a2 + 16) )
  {
LABEL_8:
    sub_C8CC70(a2, v9, a3, a4, a5, a6);
    v13 = *(unsigned __int8 *)(a2 + 28);
    v10 = *(__int64 **)(a2 + 8);
  }
  else
  {
    a4 = (unsigned int)(a4 + 1);
    *(_DWORD *)(a2 + 20) = a4;
    *v11 = v9;
    v10 = *(__int64 **)(a2 + 8);
    ++*(_QWORD *)a2;
    v13 = *(unsigned __int8 *)(a2 + 28);
  }
  v12 = *(_QWORD *)(a1 + 40);
  if ( !(_BYTE)v13 )
    goto LABEL_15;
  a4 = *(unsigned int *)(a2 + 20);
LABEL_11:
  v13 = (__int64)&v10[(unsigned int)a4];
  if ( (__int64 *)v13 == v10 )
  {
LABEL_14:
    if ( (unsigned int)a4 < *(_DWORD *)(a2 + 16) )
    {
      *(_DWORD *)(a2 + 20) = a4 + 1;
      *(_QWORD *)v13 = v12;
      ++*(_QWORD *)a2;
      goto LABEL_16;
    }
LABEL_15:
    sub_C8CC70(a2, v12, v13, a4, a5, a6);
    goto LABEL_16;
  }
  while ( *v10 != v12 )
  {
    if ( (__int64 *)v13 == ++v10 )
      goto LABEL_14;
  }
LABEL_16:
  v14 = (unsigned int)v35;
  v15 = *(_QWORD *)(a1 + 32);
  v16 = (unsigned int)v35 + 1LL;
  if ( v16 > HIDWORD(v35) )
  {
    sub_C8D5F0((__int64)&v34, v36, v16, 8u, a5, a6);
    v14 = (unsigned int)v35;
  }
  *(_QWORD *)&v34[8 * v14] = v15;
  v17 = v35 + 1;
  LODWORD(v35) = v17;
  if ( v17 )
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)&v34[8 * v17 - 8];
      v19 = *(unsigned int *)(v33 + 12);
      LODWORD(v35) = v17 - 1;
      v20 = *(unsigned int *)(v33 + 8);
      if ( v20 + 1 > v19 )
      {
        sub_C8D5F0(v33, (const void *)(v33 + 16), v20 + 1, 8u, a5, a6);
        v20 = *(unsigned int *)(v33 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v33 + 8 * v20) = v18;
      v21 = (_QWORD *)(v18 + 48);
      ++*(_DWORD *)(v33 + 8);
      v22 = *v21 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (_QWORD *)v22 != v21 )
      {
        if ( !v22 )
          BUG();
        v23 = v22 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v22 - 24) - 30 <= 0xA )
        {
          v24 = sub_B46E30(v23);
          if ( v24 )
            break;
        }
      }
LABEL_32:
      v17 = v35;
      if ( !(_DWORD)v35 )
        goto LABEL_33;
    }
    v25 = 0;
    while ( 1 )
    {
      v28 = sub_B46EC0(v23, v25);
      if ( *(_BYTE *)(a2 + 28) )
      {
        v29 = *(__int64 **)(a2 + 8);
        v27 = *(unsigned int *)(a2 + 20);
        v26 = &v29[v27];
        if ( v29 != v26 )
        {
          while ( v28 != *v29 )
          {
            if ( v26 == ++v29 )
              goto LABEL_41;
          }
          goto LABEL_31;
        }
LABEL_41:
        if ( (unsigned int)v27 < *(_DWORD *)(a2 + 16) )
        {
          *(_DWORD *)(a2 + 20) = v27 + 1;
          *v26 = v28;
          ++*(_QWORD *)a2;
          goto LABEL_37;
        }
      }
      sub_C8CC70(a2, v28, (__int64)v26, v27, a5, a6);
      if ( v30 )
      {
LABEL_37:
        v31 = (unsigned int)v35;
        v32 = (unsigned int)v35 + 1LL;
        if ( v32 > HIDWORD(v35) )
        {
          sub_C8D5F0((__int64)&v34, v36, v32, 8u, a5, a6);
          v31 = (unsigned int)v35;
        }
        ++v25;
        *(_QWORD *)&v34[8 * v31] = v28;
        LODWORD(v35) = v35 + 1;
        if ( v25 == v24 )
          goto LABEL_32;
      }
      else
      {
LABEL_31:
        if ( ++v25 == v24 )
          goto LABEL_32;
      }
    }
  }
LABEL_33:
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
}
