// Function: sub_164BAF0
// Address: 0x164baf0
//
__int64 __fastcall sub_164BAF0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // r8
  __int64 v6; // r9
  unsigned int v7; // r13d
  unsigned int v8; // edi
  __int64 *v9; // rax
  __int64 v10; // rcx
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 (__fastcall *v15)(__int64); // rax
  __int64 result; // rax
  int v17; // r11d
  __int64 *v18; // rdx
  int v19; // eax
  int v20; // ecx
  int v21; // eax
  int v22; // esi
  __int64 v23; // r8
  unsigned int v24; // eax
  __int64 v25; // rdi
  int v26; // r10d
  __int64 *v27; // r9
  int v28; // eax
  int v29; // eax
  int v30; // r9d
  __int64 *v31; // r8
  __int64 v32; // rdi
  unsigned int v33; // r13d
  __int64 v34; // rsi
  __int64 *v35; // r14
  __int64 v36; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v37; // [rsp+8h] [rbp-38h]
  __int64 v38; // [rsp+10h] [rbp-30h]

  v2 = sub_16498A0(a1);
  v3 = *(_QWORD *)v2;
  v4 = *(_DWORD *)(*(_QWORD *)v2 + 2664LL);
  v5 = *(_QWORD *)v2 + 2640LL;
  if ( v4 )
  {
    v6 = *(_QWORD *)(v3 + 2648);
    v7 = ((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4);
    v8 = (v4 - 1) & v7;
    v9 = (__int64 *)(v6 + 16LL * ((v4 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4))));
    v10 = *v9;
    if ( *v9 == a1 )
    {
      v11 = (__int64 *)v9[1];
      goto LABEL_4;
    }
    v17 = 1;
    v18 = 0;
    while ( v10 != -8 )
    {
      if ( v10 != -16 || v18 )
        v9 = v18;
      v8 = (v4 - 1) & (v17 + v8);
      v35 = (__int64 *)(v6 + 16LL * v8);
      v10 = *v35;
      if ( *v35 == a1 )
      {
        v11 = (__int64 *)v35[1];
        goto LABEL_4;
      }
      ++v17;
      v18 = v9;
      v9 = (__int64 *)(v6 + 16LL * v8);
    }
    if ( !v18 )
      v18 = v9;
    v19 = *(_DWORD *)(v3 + 2656);
    ++*(_QWORD *)(v3 + 2640);
    v20 = v19 + 1;
    if ( 4 * (v19 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(v3 + 2660) - v20 > v4 >> 3 )
        goto LABEL_27;
      sub_164B930(v5, v4);
      v28 = *(_DWORD *)(v3 + 2664);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = 1;
        v31 = 0;
        v32 = *(_QWORD *)(v3 + 2648);
        v33 = v29 & v7;
        v20 = *(_DWORD *)(v3 + 2656) + 1;
        v18 = (__int64 *)(v32 + 16LL * v33);
        v34 = *v18;
        if ( *v18 != a1 )
        {
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v31 )
              v31 = v18;
            v33 = v29 & (v30 + v33);
            v18 = (__int64 *)(v32 + 16LL * v33);
            v34 = *v18;
            if ( *v18 == a1 )
              goto LABEL_27;
            ++v30;
          }
          if ( v31 )
            v18 = v31;
        }
        goto LABEL_27;
      }
LABEL_60:
      ++*(_DWORD *)(v3 + 2656);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(v3 + 2640);
  }
  sub_164B930(v5, 2 * v4);
  v21 = *(_DWORD *)(v3 + 2664);
  if ( !v21 )
    goto LABEL_60;
  v22 = v21 - 1;
  v23 = *(_QWORD *)(v3 + 2648);
  v24 = (v21 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v20 = *(_DWORD *)(v3 + 2656) + 1;
  v18 = (__int64 *)(v23 + 16LL * v24);
  v25 = *v18;
  if ( *v18 != a1 )
  {
    v26 = 1;
    v27 = 0;
    while ( v25 != -8 )
    {
      if ( !v27 && v25 == -16 )
        v27 = v18;
      v24 = v22 & (v26 + v24);
      v18 = (__int64 *)(v23 + 16LL * v24);
      v25 = *v18;
      if ( *v18 == a1 )
        goto LABEL_27;
      ++v26;
    }
    if ( v27 )
      v18 = v27;
  }
LABEL_27:
  *(_DWORD *)(v3 + 2656) = v20;
  if ( *v18 != -8 )
    --*(_DWORD *)(v3 + 2660);
  *v18 = a1;
  v11 = 0;
  v18[1] = 0;
LABEL_4:
  v12 = v11[2];
  v36 = 0;
  v37 = 0;
  v38 = v12;
  if ( v12 != 0 && v12 != -8 && v12 != -16 )
    sub_1649AC0((unsigned __int64 *)&v36, *v11 & 0xFFFFFFFFFFFFFFF8LL);
  do
  {
    while ( 1 )
    {
      sub_1649B30(&v36);
      sub_1649AF0(&v36, (__int64)v11);
      v14 = (*v11 >> 1) & 3;
      if ( v14 == 1 )
        break;
      if ( (unsigned int)(v14 - 2) <= 1 )
        goto LABEL_8;
LABEL_13:
      v11 = v37;
      if ( !v37 )
        goto LABEL_17;
    }
    v15 = *(__int64 (__fastcall **)(__int64))(*(v11 - 1) + 8);
    if ( v15 == sub_1649C10 )
    {
LABEL_8:
      v13 = v11[2];
      if ( v13 )
      {
        if ( v13 != -8 && v13 != -16 )
          sub_1649B30(v11);
        v11[2] = 0;
      }
      goto LABEL_13;
    }
    v15((__int64)(v11 - 1));
    v11 = v37;
  }
  while ( v37 );
LABEL_17:
  result = v38;
  if ( v38 != 0 && v38 != -8 && v38 != -16 )
    return sub_1649B30(&v36);
  return result;
}
