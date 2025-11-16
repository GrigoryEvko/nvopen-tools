// Function: sub_390FF80
// Address: 0x390ff80
//
__int64 __fastcall sub_390FF80(__int64 a1, __int64 a2, unsigned __int8 *a3, size_t a4)
{
  __int64 v6; // r12
  unsigned int v7; // r10d
  __int64 *v8; // r11
  __int64 *v9; // rax
  __int64 v10; // rdx
  __int64 *v11; // rcx
  int v12; // eax
  __int64 v14; // rax
  unsigned int v15; // r10d
  __int64 *v16; // r11
  __int64 v17; // rcx
  _BYTE *v18; // rdi
  __int64 **v19; // rax
  int v20; // r8d
  int v21; // r9d
  __int64 *v22; // rdx
  __int64 **v23; // rax
  int v24; // eax
  __int64 v25; // rbx
  const void *v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rax
  size_t v29; // rbx
  __int64 v30; // rax
  _BYTE *v31; // rax
  __int64 *v32; // [rsp+0h] [rbp-60h]
  __int64 v33; // [rsp+0h] [rbp-60h]
  unsigned int v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 *v36; // [rsp+8h] [rbp-58h]
  __int64 *v37; // [rsp+10h] [rbp-50h]
  unsigned int v38; // [rsp+10h] [rbp-50h]
  unsigned int v39; // [rsp+18h] [rbp-48h]
  int v40; // [rsp+24h] [rbp-3Ch]

  v6 = sub_390FEA0(a2);
  v40 = *(_DWORD *)(v6 + 72);
  v7 = sub_16D19C0(a2 + 24, a3, a4);
  v8 = (__int64 *)(*(_QWORD *)(a2 + 24) + 8LL * v7);
  v9 = (__int64 *)*v8;
  if ( !*v8 )
  {
LABEL_6:
    v32 = v8;
    v34 = v7;
    v14 = malloc(a4 + 17);
    v15 = v34;
    v16 = v32;
    v17 = v14;
    if ( !v14 )
    {
      if ( a4 == -17 )
      {
        v30 = malloc(1u);
        v15 = v34;
        v16 = v32;
        v17 = 0;
        if ( v30 )
        {
          v18 = (_BYTE *)(v30 + 16);
          v17 = v30;
          goto LABEL_19;
        }
      }
      v33 = v17;
      v36 = v16;
      v38 = v15;
      sub_16BD1C0("Allocation failed", 1u);
      v15 = v38;
      v16 = v36;
      v17 = v33;
    }
    v18 = (_BYTE *)(v17 + 16);
    if ( a4 + 1 <= 1 )
    {
LABEL_8:
      v18[a4] = 0;
      *(_QWORD *)v17 = a4;
      *(_DWORD *)(v17 + 8) = v40;
      *v16 = v17;
      ++*(_DWORD *)(a2 + 36);
      v19 = (__int64 **)(*(_QWORD *)(a2 + 24) + 8LL * (unsigned int)sub_16D1CD0(a2 + 24, v15));
      v22 = *v19;
      if ( !*v19 || v22 == (__int64 *)-8LL )
      {
        v23 = v19 + 1;
        do
        {
          do
            v22 = *v23++;
          while ( !v22 );
        }
        while ( v22 == (__int64 *)-8LL );
      }
      v24 = *((_DWORD *)v22 + 2);
      v25 = *v22;
      v26 = v22 + 2;
      v27 = *(unsigned int *)(v6 + 72);
      *(_QWORD *)a1 = v22 + 2;
      *(_DWORD *)(a1 + 16) = v24;
      v28 = *(unsigned int *)(v6 + 76);
      *(_QWORD *)(a1 + 8) = v25;
      v29 = v25 + 1;
      if ( v29 > v28 - v27 )
      {
        sub_16CD150(v6 + 64, (const void *)(v6 + 80), v29 + v27, 1, v20, v21);
        v27 = *(unsigned int *)(v6 + 72);
        if ( !v29 )
          goto LABEL_16;
      }
      else if ( !v29 )
      {
LABEL_16:
        *(_DWORD *)(v6 + 72) = v27 + v29;
        return a1;
      }
      memcpy((void *)(*(_QWORD *)(v6 + 64) + v27), v26, v29);
      LODWORD(v27) = *(_DWORD *)(v6 + 72);
      goto LABEL_16;
    }
LABEL_19:
    v35 = v17;
    v37 = v16;
    v39 = v15;
    v31 = memcpy(v18, a3, a4);
    v17 = v35;
    v16 = v37;
    v15 = v39;
    v18 = v31;
    goto LABEL_8;
  }
  if ( v9 == (__int64 *)-8LL )
  {
    --*(_DWORD *)(a2 + 40);
    goto LABEL_6;
  }
  v10 = *v9;
  v11 = v9 + 2;
  v12 = *((_DWORD *)v9 + 2);
  *(_QWORD *)a1 = v11;
  *(_QWORD *)(a1 + 8) = v10;
  *(_DWORD *)(a1 + 16) = v12;
  return a1;
}
