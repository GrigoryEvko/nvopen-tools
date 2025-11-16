// Function: sub_1DB7AE0
// Address: 0x1db7ae0
//
void __fastcall sub_1DB7AE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 *v7; // r12
  unsigned int v8; // ebx
  __int64 v9; // rcx
  __int64 *v10; // r14
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-48h]
  unsigned int v13; // [rsp+10h] [rbp-40h]

  v12 = *(_QWORD *)(a2 + 48);
  v5 = sub_220EF30(a2);
  v6 = (a3 >> 1) & 3;
  v7 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 96LL) + 8LL);
  v13 = v6;
  if ( (__int64 *)v5 != v7 )
  {
    v8 = v6 | *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24);
    while ( v8 >= (*(_DWORD *)((*(_QWORD *)(v5 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                 | (unsigned int)(*(__int64 *)(v5 + 40) >> 1) & 3) )
    {
      v5 = sub_220EF30(v5);
      if ( (__int64 *)v5 == v7 )
        goto LABEL_6;
    }
    v7 = (__int64 *)v5;
  }
LABEL_6:
  v9 = *(_QWORD *)(sub_220EFE0(v7) + 40);
  if ( (*(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | v13) < (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                              | (unsigned int)(v9 >> 1) & 3) )
    *(_QWORD *)(a2 + 40) = v9;
  else
    *(_QWORD *)(a2 + 40) = a3;
  v10 = *(__int64 **)(*(_QWORD *)a1 + 96LL);
  if ( v7 != v10 + 1
    && (*(_DWORD *)((v7[4] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v7[4] >> 1) & 3) <= (*(_DWORD *)((*(_QWORD *)(a2 + 40) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(*(__int64 *)(a2 + 40) >> 1)
                                                                                             & 3)
    && v7[6] == v12 )
  {
    *(_QWORD *)(a2 + 40) = v7[5];
    v7 = (__int64 *)sub_220EF30(v7);
    v10 = *(__int64 **)(*(_QWORD *)a1 + 96LL);
  }
  v11 = sub_220EF30(a2);
  sub_1DB7A40(v10, v11, v7);
}
