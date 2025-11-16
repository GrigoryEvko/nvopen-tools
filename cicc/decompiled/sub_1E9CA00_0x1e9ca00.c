// Function: sub_1E9CA00
// Address: 0x1e9ca00
//
__int64 __fastcall sub_1E9CA00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _DWORD *v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r8d
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 i; // rax
  __int64 v19; // rbx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  _BYTE *v23; // r9
  __int64 v24; // r12
  int v25; // eax
  __int64 v26; // rdi
  int v27; // r8d
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rdx
  int v33; // [rsp+8h] [rbp-48h]
  int v34; // [rsp+Ch] [rbp-44h]
  int v35; // [rsp+Ch] [rbp-44h]
  int v36; // [rsp+18h] [rbp-38h] BYREF
  int v37[13]; // [rsp+1Ch] [rbp-34h] BYREF

  if ( *(_QWORD *)(a3 + 64) )
  {
    v12 = *(_QWORD *)(a3 + 40);
    v13 = a3 + 32;
    if ( v12 )
    {
      v14 = a3 + 32;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(v12 + 16);
          v16 = *(_QWORD *)(v12 + 24);
          if ( *(_DWORD *)(v12 + 32) >= a2 )
            break;
          v12 = *(_QWORD *)(v12 + 24);
          if ( !v16 )
            goto LABEL_16;
        }
        v14 = v12;
        v12 = *(_QWORD *)(v12 + 16);
      }
      while ( v15 );
LABEL_16:
      if ( v13 != v14 )
      {
        v10 = 1;
        if ( *(_DWORD *)(v14 + 32) <= a2 )
          return v10;
      }
    }
  }
  else
  {
    v8 = *(_DWORD **)a3;
    v9 = *(_QWORD *)a3 + 4LL * *(unsigned int *)(a3 + 8);
    if ( v8 != (_DWORD *)v9 )
    {
      while ( *v8 != a2 )
      {
        if ( (_DWORD *)v9 == ++v8 )
          goto LABEL_3;
      }
      v10 = 1;
      if ( (_DWORD *)v9 != v8 )
        return v10;
    }
  }
LABEL_3:
  if ( (unsigned __int8)sub_1E69E00(*(_QWORD *)(a1 + 248), a2) && (unsigned int)dword_4FC8580 > *(_DWORD *)(a4 + 8) )
  {
    v17 = *(_QWORD *)(a1 + 248);
    for ( i = (a2 & 0x80000000) != 0
            ? *(_QWORD *)(*(_QWORD *)(v17 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8)
            : *(_QWORD *)(*(_QWORD *)(v17 + 272) + 8LL * a2);
          i && ((*(_BYTE *)(i + 3) & 0x10) != 0 || (*(_BYTE *)(i + 4) & 8) != 0);
          i = *(_QWORD *)(i + 32) )
    {
      ;
    }
    v19 = *(_QWORD *)(i + 16);
    v36 = sub_1E165A0(v19, a2, 0, 0);
    if ( *(_BYTE *)(*(_QWORD *)(v19 + 16) + 4LL) == 1 )
    {
      v24 = *(_QWORD *)(v19 + 32);
      if ( !*(_BYTE *)v24
        && *(int *)(v24 + 8) < 0
        && (*(_BYTE *)(v24 + 3) & 0x10) != 0
        && (*(_WORD *)(v24 + 2) & 0xFF0) != 0 )
      {
        v25 = sub_1E16AB0(v19, 0, v20, v21, v22, v23);
        if ( v36 == v25 )
        {
          v31 = *(_DWORD *)(a4 + 8);
          if ( v31 >= *(_DWORD *)(a4 + 12) )
          {
            sub_1E9C730(a4);
            v31 = *(_DWORD *)(a4 + 8);
          }
          v32 = *(_QWORD *)a4 + 24LL * v31;
          if ( v32 )
          {
            *(_QWORD *)v32 = v19;
            *(_BYTE *)(v32 + 16) = 0;
            v31 = *(_DWORD *)(a4 + 8);
          }
          *(_DWORD *)(a4 + 8) = v31 + 1;
          return (unsigned int)sub_1E9CA00(a1, *(unsigned int *)(v24 + 8), a3, a4);
        }
        v34 = v25;
        v26 = *(_QWORD *)(a1 + 232);
        v37[0] = -1;
        if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64, int *, int *))(*(_QWORD *)v26 + 176LL))(
               v26,
               v19,
               &v36,
               v37) )
        {
          v27 = v34;
          if ( v37[0] == v34 )
          {
            v28 = v36;
            v29 = *(unsigned int *)(a4 + 8);
            if ( (unsigned int)v29 >= *(_DWORD *)(a4 + 12) )
            {
              v33 = v34;
              v35 = v36;
              sub_1E9C730(a4);
              v29 = *(unsigned int *)(a4 + 8);
              v27 = v33;
              v28 = v35;
            }
            v30 = *(_QWORD *)a4 + 24 * v29;
            if ( v30 )
            {
              *(_QWORD *)v30 = v19;
              *(_BYTE *)(v30 + 16) = 1;
              *(_DWORD *)(v30 + 8) = v28;
              *(_DWORD *)(v30 + 12) = v27;
            }
            ++*(_DWORD *)(a4 + 8);
            return (unsigned int)sub_1E9CA00(a1, *(unsigned int *)(v24 + 8), a3, a4);
          }
        }
      }
    }
  }
  return 0;
}
