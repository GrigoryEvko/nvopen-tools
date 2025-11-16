// Function: sub_2E4AD70
// Address: 0x2e4ad70
//
void __fastcall sub_2E4AD70(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, char a5)
{
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rcx
  int v11; // r13d
  __int16 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rcx
  unsigned int v15; // esi
  int *v16; // rbx
  int v17; // edi
  __int64 v18; // rsi
  __int64 v19; // rsi
  int v20; // ecx
  int *v21; // r14
  char v22; // r12
  int *v23; // r13
  __int64 v24; // rsi
  int v25; // eax
  int v26; // ecx
  int v27; // edx
  unsigned int v28; // eax
  int *v29; // rbx
  int v30; // edi
  unsigned __int64 v31; // rdi
  int v32; // r9d
  int v33; // r10d
  char v34; // [rsp+1Ch] [rbp-B4h] BYREF
  _QWORD v35[4]; // [rsp+20h] [rbp-B0h] BYREF
  int *v36; // [rsp+40h] [rbp-90h] BYREF
  __int64 v37; // [rsp+48h] [rbp-88h]
  _BYTE v38[40]; // [rsp+50h] [rbp-80h] BYREF
  int v39; // [rsp+78h] [rbp-58h] BYREF
  unsigned __int64 v40; // [rsp+80h] [rbp-50h]
  int *v41; // [rsp+88h] [rbp-48h]
  int *v42; // [rsp+90h] [rbp-40h]
  __int64 v43; // [rsp+98h] [rbp-38h]

  v36 = (int *)v38;
  v37 = 0x800000000LL;
  v41 = &v39;
  v42 = &v39;
  v35[3] = &v36;
  v35[1] = &v34;
  v8 = *(_QWORD *)(a3 + 8);
  v39 = 0;
  v40 = 0;
  v43 = 0;
  v35[2] = a3;
  v9 = *(_DWORD *)(v8 + 24LL * a2 + 16);
  v35[0] = a4;
  v10 = *(_QWORD *)(a3 + 56);
  v34 = a5;
  v11 = v9 & 0xFFF;
  v12 = (__int16 *)(v10 + 2LL * (v9 >> 12));
  do
  {
    if ( !v12 )
      break;
    v13 = *(unsigned int *)(a1 + 24);
    v14 = *(_QWORD *)(a1 + 8);
    if ( (_DWORD)v13 )
    {
      v15 = (v13 - 1) & (37 * v11);
      v16 = (int *)(v14 + ((unsigned __int64)v15 << 7));
      v17 = *v16;
      if ( v11 == *v16 )
      {
LABEL_5:
        if ( v16 != (int *)(v14 + (v13 << 7)) )
        {
          v18 = *((_QWORD *)v16 + 1);
          if ( v18 )
            sub_2E4ACA0((__int64)v35, v18);
          v19 = *((_QWORD *)v16 + 2);
          if ( v19 )
            sub_2E4ACA0((__int64)v35, v19);
        }
      }
      else
      {
        v33 = 1;
        while ( v17 != -1 )
        {
          v15 = (v13 - 1) & (v33 + v15);
          v16 = (int *)(v14 + ((unsigned __int64)v15 << 7));
          v17 = *v16;
          if ( v11 == *v16 )
            goto LABEL_5;
          ++v33;
        }
      }
    }
    v20 = *v12++;
    v11 += v20;
  }
  while ( (_WORD)v20 );
  if ( v43 )
  {
    v21 = v41;
    v23 = &v39;
    v22 = 0;
  }
  else
  {
    v21 = v36;
    v22 = 1;
    v23 = &v36[(unsigned int)v37];
  }
  while ( v23 != v21 )
  {
    v24 = *(_QWORD *)(a1 + 8);
    v25 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v26 = *v21;
      if ( !v25 )
        goto LABEL_29;
    }
    else
    {
      v26 = v21[8];
      if ( !v25 )
        goto LABEL_23;
    }
    v27 = v25 - 1;
    v28 = (v25 - 1) & (37 * v26);
    v29 = (int *)(v24 + ((unsigned __int64)v28 << 7));
    v30 = *v29;
    if ( *v29 == v26 )
    {
LABEL_17:
      v31 = *((_QWORD *)v29 + 11);
      if ( (int *)v31 != v29 + 26 )
        _libc_free(v31);
      if ( !*((_BYTE *)v29 + 52) )
        _libc_free(*((_QWORD *)v29 + 4));
      *v29 = -2;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v32 = 1;
      while ( v30 != -1 )
      {
        v28 = v27 & (v32 + v28);
        v29 = (int *)(v24 + ((unsigned __int64)v28 << 7));
        v30 = *v29;
        if ( *v29 == v26 )
          goto LABEL_17;
        ++v32;
      }
    }
    if ( v22 )
    {
LABEL_29:
      ++v21;
      continue;
    }
LABEL_23:
    v21 = (int *)sub_220EF30((__int64)v21);
  }
  sub_2E45280(v40);
  if ( v36 != (int *)v38 )
    _libc_free((unsigned __int64)v36);
}
