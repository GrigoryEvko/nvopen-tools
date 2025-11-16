// Function: sub_331BBE0
// Address: 0x331bbe0
//
__int64 __fastcall sub_331BBE0(unsigned int *a1, unsigned int a2, __int64 a3, __int64 a4)
{
  unsigned int *v4; // r10
  __int64 v8; // r14
  __int64 v9; // rdx
  unsigned int v10; // eax
  __int64 v11; // r15
  unsigned __int64 v12; // rax
  __int32 v13; // edx
  __int64 v14; // rsi
  unsigned int v15; // r14d
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r8
  unsigned int v19; // r9d
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // ecx
  unsigned int v23; // edi
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  unsigned __int64 *v30; // rcx
  __int64 v31; // rax
  _QWORD *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // r14
  __int64 v35; // rdx
  unsigned int i; // edi
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r15
  unsigned __int8 v43; // [rsp+7h] [rbp-39h]
  int v44; // [rsp+8h] [rbp-38h]
  int v45; // [rsp+8h] [rbp-38h]

  v4 = a1 + 2;
  if ( a2 != 1 )
  {
    v15 = a2;
    v19 = 0;
    v18 = a2 - 1;
    v11 = 16 * v18;
    goto LABEL_4;
  }
  v8 = *(_QWORD *)a1;
  v9 = *((_QWORD *)a1 + 1);
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 140LL);
  if ( v10 > 7 )
  {
    v11 = 16;
    v12 = sub_32B3110(*(_QWORD *)a1, *(_DWORD *)(v9 + 12));
    v13 = *(_DWORD *)(v8 + 140);
    v14 = v8 + 8;
    v15 = 2;
    sub_F038C0(a1 + 2, v14, v13, v12, v16, v17);
    v4 = a1 + 2;
    LODWORD(v18) = 1;
    v19 = 1;
LABEL_4:
    if ( !a1[4] || (v20 = *((_QWORD *)a1 + 1), *(_DWORD *)(v20 + 12) >= *(_DWORD *)(v20 + 8)) )
    {
      v43 = v19;
      v44 = v18;
      sub_F03AD0(v4, v18);
      v19 = v43;
      LODWORD(v18) = v44;
      ++*(_DWORD *)(v11 + *((_QWORD *)a1 + 1) + 12);
      v20 = *((_QWORD *)a1 + 1);
    }
    v21 = v11 + v20;
    v22 = *(_DWORD *)(v21 + 8);
    if ( v22 == 12 )
    {
      v45 = v18;
      v19 = sub_331B7E0((__int64 *)a1, v18);
      v42 = v45 + (unsigned int)(unsigned __int8)v19;
      LODWORD(v18) = v42;
      v15 = v42 + 1;
      v11 = 16 * v42;
      v21 = v11 + *((_QWORD *)a1 + 1);
      v22 = *(_DWORD *)(v21 + 8);
    }
    v23 = *(_DWORD *)(v21 + 12);
    v24 = *(_QWORD *)v21;
    v25 = v22 - 1;
    if ( v22 != v23 )
    {
      do
      {
        v26 = v25 + 1;
        *(_QWORD *)(v24 + 8 * v26) = *(_QWORD *)(v24 + 8LL * v25);
        *(_QWORD *)(v24 + 8 * v26 + 96) = *(_QWORD *)(v24 + 8LL * v25 + 96);
        LODWORD(v26) = v25--;
      }
      while ( v23 != (_DWORD)v26 );
    }
    *(_QWORD *)(v24 + 8LL * v23) = a3;
    *(_QWORD *)(v24 + 8LL * v23 + 96) = a4;
    v27 = v11 + *((_QWORD *)a1 + 1);
    v28 = *(unsigned int *)(v27 + 8);
    *(_DWORD *)(v27 + 8) = v28 + 1;
    if ( (_DWORD)v18 )
    {
      v29 = *((_QWORD *)a1 + 1) + 16LL * (unsigned int)(v18 - 1);
      v30 = (unsigned __int64 *)(*(_QWORD *)v29 + 8LL * *(unsigned int *)(v29 + 12));
      *v30 = *v30 & 0xFFFFFFFFFFFFFFC0LL | v28;
      v31 = *((_QWORD *)a1 + 1);
      v32 = (_QWORD *)(v31 + v11);
      v33 = *(unsigned int *)(v31 + v11 + 12);
      if ( (_DWORD)v33 != *(_DWORD *)(v31 + v11 + 8) - 1 )
      {
LABEL_13:
        v34 = v31 + 16LL * v15;
        v35 = *(_QWORD *)(*v32 + 8 * v33);
        *(_QWORD *)v34 = v35 & 0xFFFFFFFFFFFFFFC0LL;
        *(_DWORD *)(v34 + 8) = (v35 & 0x3F) + 1;
        return v19;
      }
      sub_325DE80((__int64)a1, v18, a4);
    }
    v31 = *((_QWORD *)a1 + 1);
    v32 = (_QWORD *)(v31 + v11);
    v33 = *(unsigned int *)(v31 + v11 + 12);
    goto LABEL_13;
  }
  for ( i = *(_DWORD *)(v9 + 12); i != v10; *(_QWORD *)(v8 + 8 * v38 + 72) = *(_QWORD *)(v8 + 8LL * v10 + 72) )
  {
    v38 = v10--;
    *(_QWORD *)(v8 + 8 * v38 + 8) = *(_QWORD *)(v8 + 8LL * v10 + 8);
  }
  v19 = 0;
  *(_QWORD *)(v8 + 8LL * i + 8) = a3;
  *(_QWORD *)(v8 + 8LL * i + 72) = a4;
  v39 = *(_DWORD *)(v8 + 140) + 1;
  *(_DWORD *)(v8 + 140) = v39;
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 8LL) = v39;
  v40 = *((_QWORD *)a1 + 1);
  v41 = *(_QWORD *)(*(_QWORD *)v40 + 8LL * *(unsigned int *)(v40 + 12));
  *(_QWORD *)(v40 + 16) = v41 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v40 + 24) = (v41 & 0x3F) + 1;
  return v19;
}
