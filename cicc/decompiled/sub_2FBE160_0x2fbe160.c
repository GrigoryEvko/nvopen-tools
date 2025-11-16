// Function: sub_2FBE160
// Address: 0x2fbe160
//
void __fastcall sub_2FBE160(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rdx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 *v21; // rax
  unsigned __int64 v22; // r12
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // eax
  unsigned int v28; // edx
  unsigned __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // r15
  __int64 v33; // r8
  __int64 v34; // r9
  unsigned __int64 v35; // r13
  unsigned __int64 v36; // r12
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 v41; // [rsp+10h] [rbp-50h]
  unsigned __int64 v42; // [rsp+10h] [rbp-50h]
  __int64 v43; // [rsp+18h] [rbp-48h] BYREF
  __int64 v44; // [rsp+28h] [rbp-38h] BYREF

  v9 = a1[1];
  v10 = *(_QWORD *)a2;
  v43 = a4;
  v11 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = 16LL * *(unsigned int *)(v10 + 24);
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v9 + 32) + 152LL) + v12);
  if ( *(_BYTE *)(a2 + 33)
    || v11
    && (*(_DWORD *)(v11 + 24) | (unsigned int)(a4 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL)
                                                                          + 24)
                                                              | (unsigned int)(*(__int64 *)(a2 + 16) >> 1) & 3) )
  {
    v14 = (__int64 *)(*(_QWORD *)(*a1 + 56) + v12);
    v15 = *v14;
    if ( (*v14 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v14[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v41 = v11;
      v16 = sub_2FB0650((_QWORD *)(*a1 + 48), *(_QWORD *)(*a1 + 40), v10, v11, v10);
      v11 = v41;
      v15 = v16;
    }
    v17 = *(_DWORD *)((*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( v11 && (v42 = v11, v18 = (a4 >> 1) & 3, ((unsigned int)v18 | *(_DWORD *)(v11 + 24)) <= (v17 | 3u)) )
    {
      sub_2FB2500((__int64)a1);
      if ( *(_BYTE *)(a2 + 33)
        && (*(_DWORD *)((*(_QWORD *)(a2 + 16) & 0xFFFFFFFFFFFFFFF8LL) + 24)
          | (unsigned int)(*(__int64 *)(a2 + 16) >> 1) & 3) >= (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                              | (unsigned int)(v15 >> 1) & 3) )
      {
        v19 = sub_2FBA8B0(a1, v15);
        v20 = *(_QWORD *)(a2 + 16);
        v44 = v19;
        sub_2FBD940((__int64)a1, v19, v20);
        v21 = &v44;
        if ( (*(_DWORD *)(v42 + 24) | (unsigned int)v18) < (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                          | (unsigned int)(v44 >> 1) & 3) )
          v21 = &v43;
        v22 = sub_2FBA5C0((__int64)a1, *v21);
        sub_2FBD6E0((__int64)(a1 + 24), v22, v44, *((unsigned int *)a1 + 20), v23, v24);
        *((_DWORD *)a1 + 20) = a3;
        sub_2FBD6E0((__int64)(a1 + 24), v13, v22, a3, v25, v26);
      }
      else
      {
        v35 = sub_2FBA740((__int64)a1, *(_QWORD *)(a2 + 16));
        v36 = sub_2FBA5C0((__int64)a1, v43);
        sub_2FBD6E0((__int64)(a1 + 24), v36, v35, *((unsigned int *)a1 + 20), v37, v38);
        *((_DWORD *)a1 + 20) = a3;
        sub_2FBD6E0((__int64)(a1 + 24), v13, v36, a3, v39, v40);
      }
    }
    else
    {
      v27 = v17 | (*(__int64 *)(a2 + 16) >> 1) & 3;
      v28 = *(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v15 >> 1) & 3;
      *((_DWORD *)a1 + 20) = a3;
      if ( v27 >= v28 )
      {
        v32 = sub_2FBA8B0(a1, v15);
        sub_2FBD940((__int64)a1, v32, *(_QWORD *)(a2 + 16));
        sub_2FBD6E0((__int64)(a1 + 24), v13, v32, *((unsigned int *)a1 + 20), v33, v34);
      }
      else
      {
        v29 = sub_2FBA740((__int64)a1, *(_QWORD *)(a2 + 16));
        sub_2FBD6E0((__int64)(a1 + 24), v13, v29, *((unsigned int *)a1 + 20), v30, v31);
      }
    }
  }
  else
  {
    *((_DWORD *)a1 + 20) = a3;
    sub_2FBD6E0((__int64)(a1 + 24), v13, *(_QWORD *)(a2 + 16), a3, v10, a6);
  }
}
