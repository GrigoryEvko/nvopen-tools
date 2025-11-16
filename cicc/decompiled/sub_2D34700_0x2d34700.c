// Function: sub_2D34700
// Address: 0x2d34700
//
__int64 __fastcall sub_2D34700(unsigned int *a1, unsigned int a2, __int64 a3, int a4)
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
  unsigned int v37; // edi
  unsigned int v38; // eax
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r15
  unsigned __int8 v44; // [rsp+7h] [rbp-39h]
  int v45; // [rsp+8h] [rbp-38h]
  int v46; // [rsp+8h] [rbp-38h]

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
  v10 = *(_DWORD *)(*(_QWORD *)a1 + 196LL);
  if ( v10 > 0xE )
  {
    v11 = 16;
    v12 = sub_2D2FBC0(*(_QWORD *)a1, *(_DWORD *)(v9 + 12));
    v13 = *(_DWORD *)(v8 + 196);
    v14 = v8 + 8;
    v15 = 2;
    sub_F038C0(a1 + 2, v14, v13, v12, v16, v17);
    v4 = a1 + 2;
    LODWORD(v18) = 1;
    v19 = 1;
LABEL_4:
    if ( !a1[4] || (v20 = *((_QWORD *)a1 + 1), *(_DWORD *)(v20 + 12) >= *(_DWORD *)(v20 + 8)) )
    {
      v44 = v19;
      v45 = v18;
      sub_F03AD0(v4, v18);
      v19 = v44;
      LODWORD(v18) = v45;
      ++*(_DWORD *)(v11 + *((_QWORD *)a1 + 1) + 12);
      v20 = *((_QWORD *)a1 + 1);
    }
    v21 = v11 + v20;
    v22 = *(_DWORD *)(v21 + 8);
    if ( v22 == 16 )
    {
      v46 = v18;
      v19 = sub_2D34300((__int64 *)a1, v18);
      v43 = v46 + (unsigned int)(unsigned __int8)v19;
      LODWORD(v18) = v43;
      v15 = v43 + 1;
      v11 = 16 * v43;
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
        *(_DWORD *)(v24 + 4 * v26 + 128) = *(_DWORD *)(v24 + 4LL * v25 + 128);
        LODWORD(v26) = v25--;
      }
      while ( v23 != (_DWORD)v26 );
    }
    *(_QWORD *)(v24 + 8LL * v23) = a3;
    *(_DWORD *)(v24 + 4LL * v23 + 128) = a4;
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
      sub_2D22A70((__int64)a1, v18, a4);
    }
    v31 = *((_QWORD *)a1 + 1);
    v32 = (_QWORD *)(v31 + v11);
    v33 = *(unsigned int *)(v31 + v11 + 12);
    goto LABEL_13;
  }
  v37 = *(_DWORD *)(v9 + 12);
  if ( v10 != v37 )
  {
    v38 = v10 - 1;
    do
    {
      v39 = v38 + 1;
      *(_QWORD *)(v8 + 8 * v39 + 8) = *(_QWORD *)(v8 + 8LL * v38 + 8);
      *(_DWORD *)(v8 + 4 * v39 + 128) = *(_DWORD *)(v8 + 4LL * v38 + 128);
      LODWORD(v39) = v38--;
    }
    while ( v37 != (_DWORD)v39 );
    v10 = *(_DWORD *)(v8 + 196);
  }
  v40 = v10 + 1;
  v19 = 0;
  *(_QWORD *)(v8 + 8LL * v37 + 8) = a3;
  *(_DWORD *)(v8 + 4LL * v37 + 128) = a4;
  *(_DWORD *)(v8 + 196) = v40;
  *(_DWORD *)(*((_QWORD *)a1 + 1) + 8LL) = v40;
  v41 = *((_QWORD *)a1 + 1);
  v42 = *(_QWORD *)(*(_QWORD *)v41 + 8LL * *(unsigned int *)(v41 + 12));
  *(_QWORD *)(v41 + 16) = v42 & 0xFFFFFFFFFFFFFFC0LL;
  *(_DWORD *)(v41 + 24) = (v42 & 0x3F) + 1;
  return v19;
}
