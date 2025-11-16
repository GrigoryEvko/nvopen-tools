// Function: sub_1F20540
// Address: 0x1f20540
//
void __fastcall sub_1F20540(_QWORD *a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 v11; // r14
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rax
  unsigned __int64 v21; // r12
  __int64 v22; // r8
  int v23; // r9d
  __int64 v24; // r8
  int v25; // r9d
  unsigned int v26; // eax
  unsigned int v27; // edx
  unsigned __int64 v28; // rax
  __int64 v29; // r8
  int v30; // r9d
  __int64 v31; // r15
  __int64 v32; // r8
  int v33; // r9d
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // r12
  __int64 v36; // r8
  int v37; // r9d
  __int64 v38; // r8
  int v39; // r9d
  unsigned __int64 v40; // [rsp+10h] [rbp-50h]
  unsigned __int64 v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h] BYREF
  __int64 v43; // [rsp+28h] [rbp-38h] BYREF

  v9 = *a2;
  v42 = a4;
  v10 = a4 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[2] + 272LL) + 392LL) + 16LL * *(unsigned int *)(v9 + 48));
  if ( *((_BYTE *)a2 + 33)
    || v10
    && (*(_DWORD *)(v10 + 24) | (unsigned int)(a4 >> 1) & 3) < (*(_DWORD *)((a2[2] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                              | (unsigned int)(a2[2] >> 1) & 3) )
  {
    v12 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)*a1 + 96LL) + 8LL * *(unsigned int *)(v9 + 48));
    v13 = (__int64 *)(*(_QWORD *)(*a1 + 56LL) + 16LL * *(unsigned int *)(v12 + 48));
    v14 = *v13;
    if ( (*v13 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v13[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v40 = v10;
      v15 = sub_1F13A50((_QWORD *)(*a1 + 48LL), *(_QWORD *)(*a1 + 40LL), v12, v10, a5, a6);
      v10 = v40;
      v14 = v15;
    }
    v16 = *(_DWORD *)((a2[2] & 0xFFFFFFFFFFFFFFF8LL) + 24);
    if ( v10 && (v41 = v10, v17 = (a4 >> 1) & 3, ((unsigned int)v17 | *(_DWORD *)(v10 + 24)) <= (v16 | 3u)) )
    {
      sub_1F15650((__int64)a1);
      if ( *((_BYTE *)a2 + 33)
        && (*(_DWORD *)((a2[2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a2[2] >> 1) & 3) >= (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                 | (unsigned int)(v14 >> 1)
                                                                                                 & 3) )
      {
        v18 = sub_1F1B330(a1, v14);
        v19 = a2[2];
        v43 = v18;
        sub_1F20330((__int64)a1, v18, v19);
        v20 = &v43;
        if ( (*(_DWORD *)(v41 + 24) | (unsigned int)v17) < (*(_DWORD *)((v43 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                          | (unsigned int)(v43 >> 1) & 3) )
          v20 = &v42;
        v21 = sub_1F1B1B0((__int64)a1, *v20);
        sub_1F1FA40((__int64)(a1 + 25), v21, v43, *((unsigned int *)a1 + 20), v22, v23);
        *((_DWORD *)a1 + 20) = a3;
        sub_1F1FA40((__int64)(a1 + 25), v11, v21, a3, v24, v25);
      }
      else
      {
        v34 = sub_1F1BC20((__int64)a1, a2[2]);
        v35 = sub_1F1B1B0((__int64)a1, v42);
        sub_1F1FA40((__int64)(a1 + 25), v35, v34, *((unsigned int *)a1 + 20), v36, v37);
        *((_DWORD *)a1 + 20) = a3;
        sub_1F1FA40((__int64)(a1 + 25), v11, v35, a3, v38, v39);
      }
    }
    else
    {
      v26 = v16 | (a2[2] >> 1) & 3;
      v27 = *(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v14 >> 1) & 3;
      *((_DWORD *)a1 + 20) = a3;
      if ( v26 >= v27 )
      {
        v31 = sub_1F1B330(a1, v14);
        sub_1F20330((__int64)a1, v31, a2[2]);
        sub_1F1FA40((__int64)(a1 + 25), v11, v31, *((unsigned int *)a1 + 20), v32, v33);
      }
      else
      {
        v28 = sub_1F1BC20((__int64)a1, a2[2]);
        sub_1F1FA40((__int64)(a1 + 25), v11, v28, *((unsigned int *)a1 + 20), v29, v30);
      }
    }
  }
  else
  {
    *((_DWORD *)a1 + 20) = a3;
    sub_1F1FA40((__int64)(a1 + 25), v11, a2[2], a3, a5, a6);
  }
}
