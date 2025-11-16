// Function: sub_22A5FB0
// Address: 0x22a5fb0
//
__int64 __fastcall sub_22A5FB0(__int64 a1)
{
  void (*v1)(void); // rax
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r12
  __int64 v8; // r15
  int v9; // eax
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  unsigned int v14; // eax
  unsigned __int64 *v15; // rdx

  v1 = *(void (**)(void))(*(_QWORD *)a1 + 96LL);
  if ( (char *)v1 == (char *)sub_22A4A50 )
    sub_22A47C0(a1 + 176);
  else
    v1();
  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_17:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_4F8144C )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_17;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_4F8144C);
  v8 = **(_QWORD **)(v7 + 176);
  v9 = *(_DWORD *)(a1 + 220);
  *(_DWORD *)(a1 + 216) = 0;
  if ( v9 )
  {
    v10 = 0;
  }
  else
  {
    sub_C8D5F0(a1 + 208, (const void *)(a1 + 224), 1u, 8u, v5, v6);
    v10 = 8LL * *(unsigned int *)(a1 + 216);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 208) + v10) = v8;
  v11 = *(__int64 **)(a1 + 208);
  ++*(_DWORD *)(a1 + 216);
  v12 = *v11;
  if ( v12 )
  {
    v13 = (unsigned int)(*(_DWORD *)(v12 + 44) + 1);
    v14 = *(_DWORD *)(v12 + 44) + 1;
  }
  else
  {
    v13 = 0;
    v14 = 0;
  }
  v15 = 0;
  if ( v14 < *(_DWORD *)(v7 + 208) )
    v15 = *(unsigned __int64 **)(*(_QWORD *)(v7 + 200) + 8 * v13);
  sub_22A5210(a1 + 176, v7 + 176, v15);
  return 0;
}
