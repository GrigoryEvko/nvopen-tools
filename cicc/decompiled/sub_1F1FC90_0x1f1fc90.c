// Function: sub_1F1FC90
// Address: 0x1f1fc90
//
__int64 __fastcall sub_1F1FC90(int *a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // r12
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 *v6; // rdx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rcx
  int *v10; // r15
  unsigned __int64 *v11; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  int v14; // r9d

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*((_QWORD *)a1 + 2) + 272LL) + 392LL) + 16LL * *(unsigned int *)(a2 + 48) + 8);
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((v2 >> 1) & 3) != 0 )
    v4 = (2LL * (int)(((v2 >> 1) & 3) - 1)) | v3;
  else
    v4 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL | 6;
  v5 = *(_QWORD *)(*((_QWORD *)a1 + 9) + 8LL);
  v6 = (__int64 *)sub_1DB3C70((__int64 *)v5, v4);
  if ( v6 == (__int64 *)(*(_QWORD *)v5 + 24LL * *(unsigned int *)(v5 + 8)) )
    return v2;
  v9 = *(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v4 >> 1) & 3;
  if ( (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3) > (unsigned int)v9 )
    return v2;
  v10 = (int *)v6[2];
  if ( !v10 )
    return v2;
  v11 = (unsigned __int64 *)sub_1F14200(
                              (_QWORD *)(*(_QWORD *)a1 + 48LL),
                              *(_QWORD *)(*(_QWORD *)a1 + 40LL),
                              a2,
                              v9,
                              v7,
                              v8);
  v12 = sub_1F1AD70(a1, a1[20], v10, v4, a2, v11);
  sub_1F1FA40((__int64)(a1 + 50), *(_QWORD *)(v12 + 8), v2, (unsigned int)a1[20], v13, v14);
  return *(_QWORD *)(v12 + 8);
}
