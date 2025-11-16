// Function: sub_2E282C0
// Address: 0x2e282c0
//
__int64 __fastcall sub_2E282C0(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v6; // rdx
  unsigned int *v7; // rbx
  unsigned int v8; // r14d
  unsigned int *v9; // rax
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  __int64 v14; // rbx
  char v15; // dl
  bool v16; // r9
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned int *v20; // r14
  __int64 i; // rax
  __int64 v22; // rdx
  bool v23; // r10
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  _QWORD *v27; // r14
  __int64 v28; // rbx
  bool v29; // r9
  _QWORD *v30; // [rsp+0h] [rbp-50h]
  char v31; // [rsp+Ch] [rbp-44h]
  _QWORD *v32; // [rsp+10h] [rbp-40h]
  char v33; // [rsp+18h] [rbp-38h]
  _QWORD *v34; // [rsp+18h] [rbp-38h]
  char v35; // [rsp+18h] [rbp-38h]

  if ( *(_QWORD *)(a2 + 184) )
  {
    v11 = sub_B996D0(a2 + 144, a3);
    v13 = v12;
    v14 = v11;
    v15 = 0;
    if ( v13 )
    {
      v16 = 1;
      if ( !v11 && v13 != a2 + 152 )
        v16 = *a3 < *(_DWORD *)(v13 + 32);
      v33 = v16;
      v14 = sub_22077B0(0x28u);
      *(_DWORD *)(v14 + 32) = *a3;
      sub_220F040(v33, v14, (_QWORD *)v13, (_QWORD *)(a2 + 152));
      v15 = 1;
      ++*(_QWORD *)(a2 + 184);
    }
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v14;
    *(_BYTE *)(a1 + 16) = v15;
  }
  else
  {
    v6 = *(unsigned int *)(a2 + 8);
    v7 = (unsigned int *)(*(_QWORD *)a2 + 4 * v6);
    if ( *(unsigned int **)a2 == v7 )
    {
      if ( v6 <= 0x1F )
      {
        v8 = *a3;
LABEL_14:
        v17 = v6 + 1;
        if ( v17 > *(unsigned int *)(a2 + 12) )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v17, 4u, a5, *(_QWORD *)a2);
          v7 = (unsigned int *)(*(_QWORD *)a2 + 4LL * *(unsigned int *)(a2 + 8));
        }
        *v7 = v8;
        v18 = *(_QWORD *)a2;
        v19 = (unsigned int)(*(_DWORD *)(a2 + 8) + 1);
        *(_DWORD *)(a2 + 8) = v19;
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v18 + 4 * v19 - 4;
        *(_BYTE *)(a1 + 16) = 1;
        return a1;
      }
      v32 = (_QWORD *)(a2 + 144);
    }
    else
    {
      v8 = *a3;
      v9 = *(unsigned int **)a2;
      while ( *v9 != v8 )
      {
        if ( v7 == ++v9 )
          goto LABEL_13;
      }
      if ( v7 != v9 )
      {
        *(_BYTE *)(a1 + 8) = 1;
        *(_QWORD *)a1 = v9;
        *(_BYTE *)(a1 + 16) = 0;
        return a1;
      }
LABEL_13:
      if ( v6 <= 0x1F )
        goto LABEL_14;
      v20 = *(unsigned int **)a2;
      v32 = (_QWORD *)(a2 + 144);
      v34 = (_QWORD *)(a2 + 152);
      for ( i = sub_B9AB10((_QWORD *)(a2 + 144), a2 + 152, *(unsigned int **)a2); ; i = sub_B9AB10(
                                                                                          v32,
                                                                                          (__int64)v34,
                                                                                          v20) )
      {
        if ( v22 )
        {
          v23 = i || (_QWORD *)v22 == v34 || *v20 < *(_DWORD *)(v22 + 32);
          v30 = (_QWORD *)v22;
          v31 = v23;
          v24 = sub_22077B0(0x28u);
          *(_DWORD *)(v24 + 32) = *v20;
          sub_220F040(v31, v24, v30, v34);
          ++*(_QWORD *)(a2 + 184);
        }
        if ( v7 == ++v20 )
          break;
      }
    }
    *(_DWORD *)(a2 + 8) = 0;
    v25 = sub_B996D0((__int64)v32, a3);
    v27 = (_QWORD *)v26;
    v28 = v25;
    if ( v26 )
    {
      v29 = 1;
      if ( !v25 && v26 != a2 + 152 )
        v29 = *a3 < *(_DWORD *)(v26 + 32);
      v35 = v29;
      v28 = sub_22077B0(0x28u);
      *(_DWORD *)(v28 + 32) = *a3;
      sub_220F040(v35, v28, v27, (_QWORD *)(a2 + 152));
      ++*(_QWORD *)(a2 + 184);
    }
    *(_BYTE *)(a1 + 8) = 0;
    *(_QWORD *)a1 = v28;
    *(_BYTE *)(a1 + 16) = 1;
  }
  return a1;
}
