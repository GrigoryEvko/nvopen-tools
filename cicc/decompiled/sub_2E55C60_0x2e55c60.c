// Function: sub_2E55C60
// Address: 0x2e55c60
//
unsigned __int64 __fastcall sub_2E55C60(__int64 a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 *v4; // r8
  _DWORD *v5; // r10
  unsigned int v6; // r12d
  __int64 v7; // r13
  __int64 *v8; // r14
  int v9; // eax
  __int64 v10; // rax
  unsigned __int64 *v11; // r12
  unsigned __int64 result; // rax
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // rcx
  unsigned int v16; // r12d
  int v17; // eax
  _DWORD *v18; // r10
  __int64 *v19; // r8
  __int64 *v20; // r13
  char v21; // al
  int v22; // edx
  char v23; // al
  int v24; // eax
  char v25; // al
  int v26; // r13d
  int v27; // [rsp+Ch] [rbp-64h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  _DWORD *v30; // [rsp+18h] [rbp-58h]
  _DWORD *v31; // [rsp+20h] [rbp-50h]
  __int64 *v33; // [rsp+20h] [rbp-50h]
  _DWORD *v34; // [rsp+20h] [rbp-50h]
  __int64 *v35; // [rsp+28h] [rbp-48h]
  unsigned int i; // [rsp+28h] [rbp-48h]
  __int64 *v37; // [rsp+28h] [rbp-48h]
  __int64 *v38[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = a3;
  v5 = a4;
  v6 = *(_DWORD *)(a1 + 128);
  if ( v6 )
  {
    v16 = v6 - 1;
    v8 = 0;
    v28 = *(_QWORD *)(a1 + 112);
    v17 = sub_2E8E920(a3);
    v27 = 1;
    v18 = a4;
    v19 = a3;
    for ( i = v16 & v17; ; i = v26 )
    {
      v30 = v18;
      v33 = v19;
      v20 = (__int64 *)(v28 + 16LL * i);
      v21 = sub_2E4F140(*v19, *v20);
      v4 = v33;
      v5 = v30;
      if ( v21 )
      {
        result = *(_QWORD *)a1;
        v13 = v20[1];
        v11 = (unsigned __int64 *)(v20 + 1);
        v14 = *(_QWORD *)(a2 + 16);
        if ( !*(_QWORD *)a1 )
          goto LABEL_7;
        goto LABEL_14;
      }
      v23 = sub_2E4F140(*v20, 0);
      v4 = v33;
      v5 = v30;
      if ( v23 )
        break;
      v25 = sub_2E4F140(*v20, -1);
      v19 = v33;
      v18 = v30;
      if ( !v8 && v25 )
        v8 = (__int64 *)(v28 + 16LL * i);
      v26 = v16 & (v27 + i);
      ++v27;
    }
    v24 = *(_DWORD *)(a1 + 120);
    v6 = *(_DWORD *)(a1 + 128);
    if ( !v8 )
      v8 = (__int64 *)(v28 + 16LL * i);
    ++*(_QWORD *)(a1 + 104);
    v7 = a1 + 104;
    v9 = v24 + 1;
    v38[0] = v8;
    if ( 4 * v9 < 3 * v6 )
    {
      if ( v6 - (v9 + *(_DWORD *)(a1 + 124)) <= v6 >> 3 )
      {
        sub_2E55AD0(a1 + 104, v6);
        sub_2E513B0(a1 + 104, v33, v38);
        v8 = v38[0];
        v5 = v30;
        v4 = v33;
        v9 = *(_DWORD *)(a1 + 120) + 1;
      }
      goto LABEL_4;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 104);
    v7 = a1 + 104;
    v38[0] = 0;
  }
  v31 = v5;
  v35 = v4;
  sub_2E55AD0(v7, 2 * v6);
  sub_2E513B0(v7, v35, v38);
  v8 = v38[0];
  v4 = v35;
  v5 = v31;
  v9 = *(_DWORD *)(a1 + 120) + 1;
LABEL_4:
  *(_DWORD *)(a1 + 120) = v9;
  if ( *v8 )
    --*(_DWORD *)(a1 + 124);
  v10 = *v4;
  v8[1] = 0;
  v11 = (unsigned __int64 *)(v8 + 1);
  *v8 = v10;
  result = *(_QWORD *)a1;
  v13 = 0;
  v14 = *(_QWORD *)(a2 + 16);
  if ( *(_QWORD *)a1 )
  {
LABEL_14:
    *(_QWORD *)a1 = *(_QWORD *)result;
  }
  else
  {
LABEL_7:
    v15 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 88) += 32LL;
    result = (v15 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_QWORD *)(a1 + 16) >= result + 32 && v15 )
    {
      *(_QWORD *)(a1 + 8) = result + 32;
      if ( !result )
      {
        MEMORY[0] = v14;
        BUG();
      }
    }
    else
    {
      v34 = v5;
      v37 = v4;
      result = sub_9D1E70(a1 + 8, 32, 32, 3);
      v5 = v34;
      v4 = v37;
    }
  }
  *(_QWORD *)(result + 16) = *v4;
  v22 = *v5;
  *(_QWORD *)result = v14;
  *(_DWORD *)(result + 24) = v22;
  *(_QWORD *)(result + 8) = v13;
  *v11 = result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
