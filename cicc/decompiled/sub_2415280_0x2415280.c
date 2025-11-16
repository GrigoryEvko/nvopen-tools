// Function: sub_2415280
// Address: 0x2415280
//
__int64 __fastcall sub_2415280(__int64 a1, __int64 a2, __int64 a3, __int16 a4)
{
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v8; // rdi
  unsigned int v10; // esi
  int v11; // r10d
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  unsigned int v14; // eax
  __int64 *v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rsi
  unsigned __int8 **v18; // rbx
  __int64 v19; // rdi
  char v20; // al
  __int64 *v21; // rcx
  unsigned __int8 *v22; // rax
  char v24; // al
  int v25; // eax
  int v26; // edx
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 v29; // rcx
  int v30; // eax
  int v31; // edx
  int v32; // eax
  int v33; // eax
  int v34; // r8d
  unsigned __int64 v35; // rdi
  __int64 *v36; // [rsp+0h] [rbp-D0h]
  __int64 v37; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int64 v38[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v39; // [rsp+20h] [rbp-B0h] BYREF
  void *v40; // [rsp+90h] [rbp-40h]

  v4 = a2;
  v5 = *(_QWORD *)(a2 + 8);
  v37 = a2;
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 8) - 15) > 1u )
    return v4;
  v8 = a1 + 416;
  v10 = *(_DWORD *)(a1 + 440);
  if ( !v10 )
  {
    ++*(_QWORD *)(a1 + 416);
    v38[0] = 0;
    goto LABEL_13;
  }
  v11 = 1;
  v12 = *(_QWORD *)(a1 + 424);
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v4 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( !v13 && v16 == -8192 )
        v13 = v15;
      v14 = (v10 - 1) & (v11 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v4 == *v15 )
        goto LABEL_4;
      ++v11;
    }
    v32 = *(_DWORD *)(a1 + 432);
    if ( !v13 )
      v13 = v15;
    ++*(_QWORD *)(a1 + 416);
    v31 = v32 + 1;
    v38[0] = (unsigned __int64)v13;
    if ( 4 * (v32 + 1) < 3 * v10 )
    {
      if ( v10 - *(_DWORD *)(a1 + 436) - v31 <= v10 >> 3 )
      {
        sub_FAA400(v8, v10);
        sub_F9D990(v8, &v37, v38);
        v4 = v37;
        v13 = (_QWORD *)v38[0];
        v31 = *(_DWORD *)(a1 + 432) + 1;
      }
LABEL_16:
      *(_DWORD *)(a1 + 432) = v31;
      if ( *v13 != -4096 )
        --*(_DWORD *)(a1 + 436);
      *v13 = v4;
      v18 = (unsigned __int8 **)(v13 + 1);
      *v18 = 0;
      goto LABEL_19;
    }
LABEL_13:
    sub_FAA400(v8, 2 * v10);
    v25 = *(_DWORD *)(a1 + 440);
    if ( v25 )
    {
      v4 = v37;
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 424);
      v28 = (v25 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
      v13 = (_QWORD *)(v27 + 16LL * v28);
      v29 = *v13;
      if ( *v13 == v37 )
      {
LABEL_15:
        v30 = *(_DWORD *)(a1 + 432);
        v38[0] = (unsigned __int64)v13;
        v31 = v30 + 1;
      }
      else
      {
        v34 = 1;
        v35 = 0;
        while ( v29 != -4096 )
        {
          if ( v29 == -8192 && !v35 )
            v35 = (unsigned __int64)v13;
          v16 = (unsigned int)(v34 + 1);
          v28 = v26 & (v34 + v28);
          v13 = (_QWORD *)(v27 + 16LL * v28);
          v29 = *v13;
          if ( v37 == *v13 )
            goto LABEL_15;
          ++v34;
        }
        if ( !v35 )
          v35 = (unsigned __int64)v13;
        v31 = *(_DWORD *)(a1 + 432) + 1;
        v38[0] = v35;
        v13 = (_QWORD *)v35;
      }
    }
    else
    {
      v33 = *(_DWORD *)(a1 + 432);
      v4 = v37;
      v13 = 0;
      v38[0] = 0;
      v31 = v33 + 1;
    }
    goto LABEL_16;
  }
LABEL_4:
  v17 = v15[1];
  v18 = (unsigned __int8 **)(v15 + 1);
  if ( v17 )
  {
    v36 = v15;
    v19 = a1 + 16;
    if ( a3 )
    {
      v20 = sub_B19DB0(v19, v17, a3 - 24);
      v21 = v36;
      if ( !v20 )
        goto LABEL_7;
      return v21[1];
    }
    v24 = sub_B19DB0(v19, v17, 0);
    v21 = v36;
    if ( v24 )
      return v21[1];
LABEL_43:
    BUG();
  }
LABEL_19:
  if ( !a3 )
    goto LABEL_43;
LABEL_7:
  sub_2412230((__int64)v38, *(_QWORD *)(a3 + 16), a3, a4, 0, v16, 0, 0);
  v22 = sub_2411210(a1, v37, (__int64)v38);
  *v18 = v22;
  v4 = (__int64)v22;
  nullsub_61();
  v40 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v38[0] != &v39 )
    _libc_free(v38[0]);
  return v4;
}
