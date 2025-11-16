// Function: sub_2C097A0
// Address: 0x2c097a0
//
void __fastcall sub_2C097A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // rcx
  int v10; // edx
  unsigned int v11; // edi
  __int64 *v12; // rax
  __int64 v13; // r8
  __int64 v14; // r8
  unsigned int v15; // edi
  __int64 *v16; // rax
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rbx
  __int64 v23; // rbx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 v29; // rcx
  __int64 *v30; // rbx
  __int64 *v31; // r15
  __int64 v32; // rax
  __int64 v33; // r12
  int v34; // eax
  int v35; // eax
  int v36; // r9d
  int v37; // r10d
  __int64 v38; // [rsp+8h] [rbp-58h]
  __int64 *v39; // [rsp+10h] [rbp-50h] BYREF
  __int64 v40; // [rsp+18h] [rbp-48h]
  _BYTE v41[64]; // [rsp+20h] [rbp-40h] BYREF

  v6 = sub_AA54C0(a3);
  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(_DWORD *)(v7 + 24);
  v9 = *(_QWORD *)(v7 + 8);
  if ( !v8 )
    goto LABEL_12;
  v10 = v8 - 1;
  v11 = v10 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( a3 == *v12 )
  {
LABEL_3:
    v14 = v12[1];
  }
  else
  {
    v34 = 1;
    while ( v13 != -4096 )
    {
      v36 = v34 + 1;
      v11 = v10 & (v34 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        goto LABEL_3;
      v34 = v36;
    }
    v14 = 0;
  }
  if ( !v6 )
    goto LABEL_12;
  v15 = v10 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v16 = (__int64 *)(v9 + 16LL * v15);
  v17 = *v16;
  if ( v6 == *v16 )
  {
LABEL_6:
    v18 = v16[1];
  }
  else
  {
    v35 = 1;
    while ( v17 != -4096 )
    {
      v37 = v35 + 1;
      v15 = v10 & (v35 + v15);
      v16 = (__int64 *)(v9 + 16LL * v15);
      v17 = *v16;
      if ( v6 == *v16 )
        goto LABEL_6;
      v35 = v37;
    }
    v18 = 0;
  }
  if ( v18 != v14 )
  {
    v19 = sub_2C08E10(a1, v6);
    v21 = *(unsigned int *)(a2 + 64);
    v22 = *(_QWORD *)(v19 + 48);
    if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 68) )
    {
      sub_C8D5F0(a2 + 56, (const void *)(a2 + 72), v21 + 1, 8u, v21 + 1, v20);
      v21 = *(unsigned int *)(a2 + 64);
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * v21) = v22;
    ++*(_DWORD *)(a2 + 64);
  }
  else
  {
LABEL_12:
    v23 = *(_QWORD *)(a3 + 16);
    v39 = (__int64 *)v41;
    v40 = 0x200000000LL;
    if ( v23 )
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          return;
      }
      while ( 1 )
      {
        v25 = sub_2C08E10(a1, *(_QWORD *)(v24 + 40));
        v27 = (unsigned int)v40;
        v28 = (unsigned int)v40 + 1LL;
        if ( v28 > HIDWORD(v40) )
        {
          v38 = v25;
          sub_C8D5F0((__int64)&v39, v41, (unsigned int)v40 + 1LL, 8u, v28, v26);
          v27 = (unsigned int)v40;
          v25 = v38;
        }
        v39[v27] = v25;
        v29 = (unsigned int)(v40 + 1);
        LODWORD(v40) = v40 + 1;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          break;
        while ( 1 )
        {
          v24 = *(_QWORD *)(v23 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
            break;
          v23 = *(_QWORD *)(v23 + 8);
          if ( !v23 )
            goto LABEL_19;
        }
      }
LABEL_19:
      v30 = v39;
      v31 = &v39[v29];
      if ( v39 != v31 )
      {
        v32 = *(unsigned int *)(a2 + 64);
        do
        {
          v33 = *v30;
          if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 68) )
          {
            sub_C8D5F0(a2 + 56, (const void *)(a2 + 72), v32 + 1, 8u, v28, v26);
            v32 = *(unsigned int *)(a2 + 64);
          }
          ++v30;
          *(_QWORD *)(*(_QWORD *)(a2 + 56) + 8 * v32) = v33;
          v32 = (unsigned int)(*(_DWORD *)(a2 + 64) + 1);
          *(_DWORD *)(a2 + 64) = v32;
        }
        while ( v31 != v30 );
        v31 = v39;
      }
      if ( v31 != (__int64 *)v41 )
        _libc_free((unsigned __int64)v31);
    }
  }
}
