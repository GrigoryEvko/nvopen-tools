// Function: sub_16151B0
// Address: 0x16151b0
//
__int64 __fastcall sub_16151B0(__int64 a1, _QWORD *a2, const char *a3, size_t a4, int a5)
{
  _QWORD *v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r13
  __int64 result; // rax
  __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  __int64 v18; // rsi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // r10
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r13
  int v30; // edx
  int v31; // r14d
  int v32; // eax
  int v33; // r9d
  __int64 v34; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v35[12]; // [rsp+10h] [rbp-60h] BYREF

  sub_160F160(a1, (__int64)a2, 2, a5, a3, a4);
  sub_16C6860(v35);
  v35[2] = a2;
  v35[3] = 0;
  v35[4] = 0;
  v35[0] = &unk_49ED7C0;
  v7 = sub_1612E30(a2);
  if ( v7 )
  {
    v8 = v7;
    sub_16D7910(v7);
    sub_1403F30(&v34, a2, *(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(_QWORD *))(*a2 + 96LL))(a2);
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 48LL))(v34);
    sub_16D7950(v8, a2, v9);
  }
  else
  {
    sub_1403F30(&v34, a2, *(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(_QWORD *))(*a2 + 96LL))(a2);
    if ( v34 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v34 + 48LL))(v34);
  }
  v35[0] = &unk_49ED7C0;
  nullsub_616(v35, a2, v10, v11, v12);
  v13 = a2[2];
  result = sub_1614F20(*(_QWORD *)(a1 + 16), v13);
  v15 = result;
  if ( result )
  {
    v16 = *(_DWORD *)(a1 + 248);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(a1 + 232);
      v19 = (v16 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v20 = (__int64 *)(v18 + 16LL * v19);
      v21 = *v20;
      if ( v13 == *v20 )
      {
LABEL_8:
        *v20 = -8;
        --*(_DWORD *)(a1 + 240);
        ++*(_DWORD *)(a1 + 244);
      }
      else
      {
        v32 = 1;
        while ( v21 != -4 )
        {
          v33 = v32 + 1;
          v19 = v17 & (v32 + v19);
          v20 = (__int64 *)(v18 + 16LL * v19);
          v21 = *v20;
          if ( v13 == *v20 )
            goto LABEL_8;
          v32 = v33;
        }
      }
    }
    v22 = *(_QWORD *)(v15 + 48);
    result = (*(_QWORD *)(v15 + 56) - v22) >> 3;
    if ( (_DWORD)result )
    {
      v23 = 0;
      v24 = 8LL * (unsigned int)(result - 1);
      while ( 1 )
      {
        result = *(unsigned int *)(a1 + 248);
        if ( !(_DWORD)result )
          goto LABEL_11;
        v25 = *(_QWORD *)(a1 + 232);
        v26 = *(_QWORD *)(*(_QWORD *)(v22 + v23) + 32LL);
        v27 = (result - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v26 != *v28 )
          break;
LABEL_15:
        result = v25 + 16 * result;
        if ( v28 == (__int64 *)result || (_QWORD *)v28[1] != a2 )
          goto LABEL_11;
        *v28 = -8;
        --*(_DWORD *)(a1 + 240);
        ++*(_DWORD *)(a1 + 244);
        if ( v24 == v23 )
          return result;
LABEL_12:
        v22 = *(_QWORD *)(v15 + 48);
        v23 += 8;
      }
      v30 = 1;
      while ( v29 != -4 )
      {
        v31 = v30 + 1;
        v27 = (result - 1) & (v30 + v27);
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v26 == *v28 )
          goto LABEL_15;
        v30 = v31;
      }
LABEL_11:
      if ( v24 == v23 )
        return result;
      goto LABEL_12;
    }
  }
  return result;
}
