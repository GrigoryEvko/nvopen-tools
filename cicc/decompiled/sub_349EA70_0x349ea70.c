// Function: sub_349EA70
// Address: 0x349ea70
//
__int64 __fastcall sub_349EA70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 *v4; // rcx
  int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  int v10; // ecx
  __int64 v12; // rdx
  _DWORD v13[3]; // [rsp+Ch] [rbp-14h] BYREF

  v3 = *(_QWORD *)(a3 + 48);
  v4 = (__int64 *)(v3 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v3 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_10;
  v6 = v3 & 7;
  if ( v6 )
  {
    if ( v6 == 3 )
    {
      v4 = (__int64 *)v4[2];
      goto LABEL_4;
    }
LABEL_10:
    BUG();
  }
  *(_QWORD *)(a3 + 48) = v4;
LABEL_4:
  v7 = *v4;
  if ( !*v4 || (v7 & 4) == 0 )
    BUG();
  v8 = *(_QWORD *)(a2 + 24);
  v13[0] = 0;
  v9 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _DWORD *))(*(_QWORD *)v8 + 224LL))(
         v8,
         *(_QWORD *)(*(_QWORD *)(a3 + 24) + 32LL),
         *(unsigned int *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 16),
         v13);
  v10 = v13[0];
  *(_QWORD *)(a1 + 8) = v9;
  *(_DWORD *)a1 = v10;
  *(_QWORD *)(a1 + 16) = v12;
  return a1;
}
