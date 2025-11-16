// Function: sub_2F41AD0
// Address: 0x2f41ad0
//
char __fastcall sub_2F41AD0(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r8
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rax
  int *v15; // r9
  int v16; // esi
  __int64 v17; // rcx
  unsigned int v18; // r9d
  __int64 v19; // rcx
  _DWORD *v20; // r10
  __int64 *v21; // rax
  unsigned int v22; // ebx
  __int64 v23; // rax
  bool v24; // si
  bool v25; // cf
  bool v26; // zf
  char result; // al
  bool v28; // cl
  char v29; // si
  char v30; // cl
  char v31; // di
  _DWORD *v32; // [rsp+0h] [rbp-50h]
  int *v33; // [rsp+8h] [rbp-48h]
  unsigned int v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  unsigned __int64 v37; // [rsp+18h] [rbp-38h]
  unsigned __int64 v38; // [rsp+18h] [rbp-38h]

  v6 = a1[1];
  v7 = *(_QWORD *)(*a1 + 32LL);
  v8 = *(_QWORD *)(v6 + 32);
  v9 = v6 + 32;
  v10 = v7 + 40LL * a2;
  v11 = v7 + 40LL * a3;
  v12 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 56LL);
  v13 = *(_QWORD *)(v12 + 16LL * (*(_DWORD *)(v10 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v14 = *(_QWORD *)(v12 + 16LL * (*(_DWORD *)(v11 + 8) & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v15 = (int *)(v8 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL));
  v16 = *v15;
  if ( *(_DWORD *)(v6 + 40) != *v15 )
  {
    v33 = (int *)(v8 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL));
    v35 = v11;
    v37 = v14;
    sub_2F60630(v6 + 32, v13, v11, v6);
    v17 = a1[1];
    v15 = v33;
    v11 = v35;
    v14 = v37;
    v8 = *(_QWORD *)(v17 + 32);
    v16 = *(_DWORD *)(v17 + 40);
    v9 = v17 + 32;
  }
  v18 = v15[1];
  v19 = *(unsigned __int16 *)(*(_QWORD *)v14 + 24LL);
  v20 = (_DWORD *)(v8 + 24 * v19);
  if ( *v20 != v16 )
  {
    v32 = (_DWORD *)(v8 + 24 * v19);
    v34 = v18;
    v36 = v11;
    v38 = v14;
    sub_2F60630(v9, v14, v11, v19);
    v20 = v32;
    v18 = v34;
    v11 = v36;
    v19 = *(unsigned __int16 *)(*(_QWORD *)v38 + 24LL);
  }
  v21 = (__int64 *)a1[2];
  v22 = v20[1];
  v23 = *v21;
  v24 = *(_DWORD *)(v23 + 4LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL)) > v18;
  v25 = *(_DWORD *)(v23 + 4 * v19) < v22;
  v26 = *(_DWORD *)(v23 + 4 * v19) == v22;
  result = 1;
  v28 = !v25 && !v26;
  if ( (unsigned __int8)v24 <= (unsigned __int8)v28 )
  {
    result = 0;
    if ( (unsigned __int8)v24 >= (unsigned __int8)v28 )
    {
      v29 = *(_BYTE *)(v11 + 4);
      v30 = *(_BYTE *)(v10 + 4);
      v31 = v29 & 4;
      if ( (v30 & 4) != 0 || (*(_WORD *)(v10 + 2) & 0xFF0) != 0 )
      {
        if ( v31 )
          return a2 < a3;
        if ( (*(_WORD *)(v11 + 2) & 0xFF0) != 0 )
          return a2 < a3;
        result = 1;
        if ( (*(_DWORD *)v11 & 0xFFF00) == 0 )
        {
          result = v29 & 1;
          if ( (v29 & 1) == 0 )
            return a2 < a3;
        }
      }
      else if ( (*(_DWORD *)v10 & 0xFFF00) != 0 )
      {
        if ( !v31 && (*(_WORD *)(v11 + 2) & 0xFF0) == 0 )
        {
          if ( (*(_DWORD *)v11 & 0xFFF00) != 0 )
            return a2 < a3;
          result = v29 & 1;
          if ( (v29 & 1) != 0 )
            return a2 < a3;
        }
      }
      else
      {
        result = !(v30 & 1);
        if ( v31 || (*(_WORD *)(v11 + 2) & 0xFF0) != 0 || (*(_DWORD *)v11 & 0xFFF00) == 0 && (v29 & 1) == 0 )
        {
          if ( (v30 & 1) != 0 )
            return result;
          return a2 < a3;
        }
        if ( (v30 & 1) != 0 )
          return a2 < a3;
      }
    }
  }
  return result;
}
