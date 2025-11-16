// Function: sub_1F4CDE0
// Address: 0x1f4cde0
//
void __fastcall sub_1F4CDE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  int v3; // r8d
  int v4; // r9d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  v2 = a2 + 144;
  sub_1636A10(a2, a2);
  v5 = *(unsigned int *)(a2 + 152);
  if ( (unsigned int)v5 >= *(_DWORD *)(a2 + 156) )
  {
    sub_16CD150(v2, (const void *)(a2 + 160), 0, 8, v3, v4);
    v5 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v5) = &unk_4F96DB4;
  v6 = (unsigned int)(*(_DWORD *)(a2 + 152) + 1);
  *(_DWORD *)(a2 + 152) = v6;
  if ( *(_DWORD *)(a2 + 156) <= (unsigned int)v6 )
  {
    sub_16CD150(v2, (const void *)(a2 + 160), 0, 8, v3, v4);
    v6 = *(unsigned int *)(a2 + 152);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 144) + 8 * v6) = &unk_4FC4534;
  v7 = *(unsigned int *)(a2 + 120);
  ++*(_DWORD *)(a2 + 152);
  if ( (unsigned int)v7 >= *(_DWORD *)(a2 + 124) )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v3, v4);
    v7 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v7) = &unk_4FC4534;
  v8 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v8;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v8 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v3, v4);
    v8 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v8) = &unk_4FCA82C;
  v9 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v9;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v9 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v3, v4);
    v9 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v9) = &unk_4FC450C;
  v10 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v10;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v10 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v3, v4);
    v10 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v10) = &unk_4FC6A0C;
  v11 = (unsigned int)(*(_DWORD *)(a2 + 120) + 1);
  *(_DWORD *)(a2 + 120) = v11;
  if ( *(_DWORD *)(a2 + 124) <= (unsigned int)v11 )
  {
    sub_16CD150(a2 + 112, (const void *)(a2 + 128), 0, 8, v3, v4);
    v11 = *(unsigned int *)(a2 + 120);
  }
  *(_QWORD *)(*(_QWORD *)(a2 + 112) + 8 * v11) = &unk_4FC62EC;
  ++*(_DWORD *)(a2 + 120);
  sub_1E11F70(a1, a2);
}
