// Function: sub_199A990
// Address: 0x199a990
//
__int64 __fastcall sub_199A990(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        int a6,
        __m128i a7,
        __m128i a8)
{
  __int16 v9; // ax
  _QWORD *v13; // r10
  unsigned int v14; // ebx
  _QWORD *v15; // r15
  __int64 v16; // rax
  _QWORD *v17; // r8
  int v18; // r9d
  __int64 v19; // rdx
  __int64 *v21; // rax
  __int64 v22; // r10
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 *v26; // rbx
  __int64 v27; // rcx
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // [rsp+8h] [rbp-68h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  _QWORD *v34; // [rsp+10h] [rbp-60h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+18h] [rbp-58h]
  __int64 *v38[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v39[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a6 == 3 )
    return a1;
  v9 = *(_WORD *)(a1 + 24);
  if ( v9 == 4 )
  {
    v13 = *(_QWORD **)(a1 + 32);
    v14 = a6 + 1;
    v15 = v13;
    v34 = &v13[*(_QWORD *)(a1 + 40)];
    if ( v13 != v34 )
    {
      do
      {
        v16 = sub_199A990(*v15, a2, a3, a4, a5, v14);
        if ( v16 )
        {
          if ( a2 )
          {
            v39[1] = v16;
            v38[0] = v39;
            v39[0] = a2;
            v38[1] = (__int64 *)0x200000002LL;
            v16 = sub_147EE30(a5, v38, 0, 0, a7, a8);
            v17 = v39;
            if ( v38[0] != v39 )
            {
              v32 = v16;
              _libc_free((unsigned __int64)v38[0]);
              v16 = v32;
            }
          }
          v19 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v19 >= *(_DWORD *)(a3 + 12) )
          {
            v33 = v16;
            sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, (int)v17, v18);
            v19 = *(unsigned int *)(a3 + 8);
            v16 = v33;
          }
          *(_QWORD *)(*(_QWORD *)a3 + 8 * v19) = v16;
          ++*(_DWORD *)(a3 + 8);
        }
        ++v15;
      }
      while ( v34 != v15 );
    }
    return 0;
  }
  if ( v9 == 7 )
  {
    if ( sub_14560B0(**(_QWORD **)(a1 + 32)) || *(_QWORD *)(a1 + 40) != 2 )
      return a1;
    v25 = sub_199A990(**(_QWORD **)(a1 + 32), a2, a3, a4, a5, (unsigned int)(a6 + 1));
    v26 = (__int64 *)v25;
    if ( v25 )
    {
      v27 = *(_QWORD *)(a1 + 48);
      if ( v27 == a4 || *(_WORD *)(v25 + 24) != 7 )
      {
        if ( a2 )
          v26 = (__int64 *)sub_13A5B60((__int64)a5, a2, v25, 0, 0);
        v38[0] = v26;
        sub_1458920(a3, v38);
        v28 = **(_QWORD **)(a1 + 32);
        if ( !v28 )
          return a1;
        goto LABEL_30;
      }
      if ( v25 != **(_QWORD **)(a1 + 32) )
        goto LABEL_31;
    }
    else
    {
      v28 = **(_QWORD **)(a1 + 32);
      if ( v28 )
      {
LABEL_30:
        v29 = sub_1456040(v28);
        v30 = sub_145CF80((__int64)a5, v29, 0, 0);
        v27 = *(_QWORD *)(a1 + 48);
        v26 = (__int64 *)v30;
LABEL_31:
        v37 = v27;
        v31 = sub_13A5BC0((_QWORD *)a1, (__int64)a5);
        return sub_14799E0((__int64)a5, (__int64)v26, v31, v37, 0);
      }
    }
    return a1;
  }
  if ( v9 != 5 )
    return a1;
  if ( *(_QWORD *)(a1 + 40) != 2 )
    return a1;
  v21 = *(__int64 **)(a1 + 32);
  v22 = *v21;
  if ( *(_WORD *)(*v21 + 24) )
    return a1;
  if ( a2 )
  {
    v22 = sub_13A5B60((__int64)a5, a2, *v21, 0, 0);
    v21 = *(__int64 **)(a1 + 32);
  }
  v23 = a4;
  v36 = v22;
  v24 = sub_199A990(v21[1], v22, a3, v23, a5, (unsigned int)(a6 + 1));
  if ( !v24 )
    return 0;
  v38[0] = (__int64 *)sub_13A5B60((__int64)a5, v36, v24, 0, 0);
  sub_1458920(a3, v38);
  return 0;
}
