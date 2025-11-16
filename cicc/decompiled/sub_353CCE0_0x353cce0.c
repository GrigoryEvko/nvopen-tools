// Function: sub_353CCE0
// Address: 0x353cce0
//
__int64 __fastcall sub_353CCE0(__int64 a1, __int64 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r14d
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rbx
  const void *v8; // r14
  size_t v9; // r15
  int v10; // eax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // r14
  unsigned __int64 v15; // rdi
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned __int64 v23; // r9
  unsigned int i; // ebx
  __int64 v26; // rdx
  __int64 v27; // rcx
  unsigned __int64 *v28; // rdi
  __int64 v29; // r8
  unsigned __int64 v30; // r9
  __int64 v31; // [rsp+8h] [rbp-48h]
  unsigned int v32[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = sub_BB92D0(a1, a2);
  if ( (_BYTE)v3 )
  {
    return 0;
  }
  else
  {
    v4 = v3;
    if ( a2 + 3 != (__int64 *)(a2[3] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( !(_BYTE)qword_503D908 )
      {
        v5 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F8780C);
        if ( !v5 )
          goto LABEL_37;
        v6 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v5 + 104LL))(v5, &unk_4F8780C);
        if ( !v6 )
          goto LABEL_37;
        v7 = *(_QWORD *)(v6 + 176);
        if ( !v7 )
          goto LABEL_37;
        v8 = (const void *)a2[21];
        v9 = a2[22];
        v31 = *(_QWORD *)(v7 + 48) + 8LL * *(unsigned int *)(v7 + 56);
        v10 = sub_C92610();
        v11 = sub_C92860((__int64 *)(v7 + 48), v8, v9, v10);
        v12 = v11 == -1 ? *(_QWORD *)(v7 + 48) + 8LL * *(unsigned int *)(v7 + 56) : *(_QWORD *)(v7 + 48) + 8LL * v11;
        if ( v31 != v12 )
        {
LABEL_37:
          if ( *(_BYTE *)(sub_3111D40() + 16) )
          {
            *(_DWORD *)(a1 + 208) = 2;
            v13 = sub_22077B0(0x48u);
            if ( v13 )
            {
              *(_QWORD *)(v13 + 64) = 0;
              *(_OWORD *)(v13 + 48) = 0;
              *(_QWORD *)(v13 + 16) = v13 + 64;
              *(_QWORD *)(v13 + 24) = 1;
              *(_DWORD *)(v13 + 48) = 1065353216;
              *(_QWORD *)(v13 + 56) = 0;
              *(_OWORD *)v13 = 0;
              *(_OWORD *)(v13 + 32) = 0;
            }
            v14 = *(_QWORD *)(a1 + 200);
            *(_QWORD *)(a1 + 200) = v13;
            if ( v14 )
            {
              sub_3112140(v14 + 16);
              v15 = *(_QWORD *)(v14 + 16);
              if ( v15 != v14 + 64 )
                j_j___libc_free_0(v15);
              j_j___libc_free_0(v14);
            }
          }
          else
          {
            v28 = *(unsigned __int64 **)sub_3111D40();
            if ( v28 && sub_3114900(v28, 0, v26, v27, v29, v30) != 1 )
              *(_DWORD *)(a1 + 208) = 1;
          }
        }
      }
      v16 = *(__int64 **)(a1 + 8);
      v17 = *v16;
      v18 = v16[1];
      if ( v17 == v18 )
LABEL_34:
        BUG();
      while ( *(_UNKNOWN **)v17 != &unk_50208C0 )
      {
        v17 += 16;
        if ( v18 == v17 )
          goto LABEL_34;
      }
      v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(
              *(_QWORD *)(v17 + 8),
              &unk_50208C0);
      v32[0] = 0;
      *(_QWORD *)(a1 + 176) = v19 + 176;
      *(_DWORD *)(a1 + 188) = 0;
      v4 = sub_3539E80(a1, a2, v32);
      if ( (_BYTE)v4 )
      {
        for ( i = 0; (unsigned int)qword_503DBA8 > i; ++i )
        {
          v32[0] = 0;
          ++*(_DWORD *)(a1 + 188);
          if ( !(unsigned __int8)sub_3539E80(a1, a2, v32) )
            break;
        }
        if ( *(_DWORD *)(a1 + 208) == 2 )
          sub_3534D30(a1, (__int64)a2, v20, v21, v22, v23);
      }
    }
  }
  return v4;
}
